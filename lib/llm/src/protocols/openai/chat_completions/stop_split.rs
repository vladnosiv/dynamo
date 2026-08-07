// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Some backends emit a final chunk that carries both delta payload and
//! `finish_reason: stop` on the same choice. Clients that stop reading a
//! choice at its finish reason lose that payload, so this splits such a
//! chunk into a payload chunk followed by a terminal one.

use dynamo_protocols::types::{ChatChoiceStream, ChatCompletionStreamResponseDelta, FinishReason};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};

use super::NvCreateChatCompletionStreamResponse;

#[allow(deprecated)]
fn choice_has_delta_payload(choice: &ChatChoiceStream) -> bool {
    choice.delta.role.is_some()
        || choice.delta.content.is_some()
        || choice.delta.tool_calls.is_some()
        || choice.delta.function_call.is_some()
        || choice.delta.refusal.is_some()
        || choice.delta.reasoning_content.is_some()
}

pub fn split_merged_stop_response(
    mut response: Annotated<NvCreateChatCompletionStreamResponse>,
) -> Vec<Annotated<NvCreateChatCompletionStreamResponse>> {
    let Some(chat_response) = response.data.as_mut() else {
        return vec![response];
    };

    let mut terminal_choices = Vec::new();
    for choice in &mut chat_response.inner.choices {
        if choice.finish_reason != Some(FinishReason::Stop) || !choice_has_delta_payload(choice) {
            continue;
        }

        #[allow(deprecated)]
        terminal_choices.push(ChatChoiceStream {
            index: choice.index,
            delta: ChatCompletionStreamResponseDelta {
                role: None,
                content: None,
                tool_calls: None,
                function_call: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: choice.finish_reason,
            logprobs: None,
        });

        choice.finish_reason = None;
    }

    if terminal_choices.is_empty() {
        return vec![response];
    }

    let mut terminal_response = response.clone();
    {
        let terminal_data = terminal_response.data.as_mut().unwrap();
        terminal_data.inner.choices = terminal_choices;
        terminal_data.llm_metrics = None;
    }
    response.data.as_mut().unwrap().inner.usage = None;

    vec![response, terminal_response]
}

pub fn apply_stream<S>(
    stream: S,
) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send,
{
    stream.flat_map(|response| futures::stream::iter(split_merged_stop_response(response)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_protocols::types::{
        ChatCompletionMessageContent, CompletionUsage, CreateChatCompletionStreamResponse,
    };

    fn make_response(
        content: Option<&str>,
        finish_reason: Option<FinishReason>,
        with_usage: bool,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        #[allow(deprecated)]
        let choice = ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                role: None,
                content: content.map(|text| ChatCompletionMessageContent::Text(text.to_string())),
                tool_calls: None,
                function_call: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason,
            logprobs: None,
        };

        let usage = with_usage.then(|| CompletionUsage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        });

        Annotated {
            data: Some(NvCreateChatCompletionStreamResponse {
                inner: CreateChatCompletionStreamResponse {
                    id: "test_id".to_string(),
                    choices: vec![choice],
                    created: 1234567890,
                    model: "test-model".to_string(),
                    service_tier: None,
                    system_fingerprint: None,
                    object: "chat.completion.chunk".to_string(),
                    usage,
                },
                nvext: None,
                llm_metrics: None,
            }),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
            error: None,
        }
    }

    #[test]
    fn merged_stop_and_content_splits_into_two_chunks() {
        let response = make_response(Some("hello"), Some(FinishReason::Stop), true);
        let parts = split_merged_stop_response(response);

        assert_eq!(parts.len(), 2);

        let content_chunk = parts[0].data.as_ref().unwrap();
        assert_eq!(
            content_chunk.inner.choices[0].delta.content,
            Some(ChatCompletionMessageContent::Text("hello".to_string()))
        );
        assert_eq!(content_chunk.inner.choices[0].finish_reason, None);
        assert!(content_chunk.inner.usage.is_none());

        let terminal_chunk = parts[1].data.as_ref().unwrap();
        let terminal_choice = &terminal_chunk.inner.choices[0];
        assert_eq!(terminal_choice.finish_reason, Some(FinishReason::Stop));
        assert!(terminal_choice.delta.content.is_none());
        assert!(terminal_choice.delta.tool_calls.is_none());
        assert_eq!(
            terminal_chunk.inner.usage.as_ref().unwrap().total_tokens,
            15
        );
    }

    #[test]
    fn pure_stop_chunk_passes_through_unsplit() {
        let response = make_response(None, Some(FinishReason::Stop), false);
        let parts = split_merged_stop_response(response);

        assert_eq!(parts.len(), 1);
        assert_eq!(
            parts[0].data.as_ref().unwrap().inner.choices[0].finish_reason,
            Some(FinishReason::Stop)
        );
    }

    #[test]
    fn content_chunk_without_finish_passes_through_unsplit() {
        let response = make_response(Some("hello"), None, false);
        let parts = split_merged_stop_response(response);

        assert_eq!(parts.len(), 1);
    }

    #[test]
    fn non_stop_finish_reason_with_payload_is_not_split() {
        let response = make_response(Some("hello"), Some(FinishReason::Length), false);
        let parts = split_merged_stop_response(response);

        assert_eq!(parts.len(), 1);
        assert_eq!(
            parts[0].data.as_ref().unwrap().inner.choices[0].finish_reason,
            Some(FinishReason::Length)
        );
    }
}
