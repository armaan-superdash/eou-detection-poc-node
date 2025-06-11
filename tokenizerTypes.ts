import { Tensor } from "@huggingface/transformers";

export interface TMessage {
    role: string;
    content: string;
};

type BatchEncodingItem = number[] | number[][] | Tensor;

export interface BatchEncoding {
    /** List of token ids to be fed to a model. */
    input_ids: BatchEncodingItem;

    /** List of indices specifying which tokens should be attended to by the model. */
    attention_mask: BatchEncodingItem;

    /** List of token type ids to be fed to a model. */
    token_type_ids?: BatchEncodingItem;
}
