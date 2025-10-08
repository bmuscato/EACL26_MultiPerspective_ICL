
#Compute token attributions and predictions for given text prompts
#using Integrated Gradients on a causal language model.


import os
import torch
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from captum.attr import IntegratedGradients


class AttributionAnalyzer:
    def __init__(self, model_name: str):
        """
        Initialize tokenizer, model, and device.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()


    def clean_and_reconstruct_tokens(tokens):
        """
        Reconstruct readable tokens by removing special ones and joining subwords.
        """
        cleaned_tokens = []
        skip_tokens = {
            "<s>", "</s>", "[CLS]", "[SEP]", "[PAD]", "[UNK]",
            "<bos>", "<eos>", "<begin_of_sentence>",
            "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
        }

        current_word = ""
        for token in tokens:
            if token in skip_tokens:
                continue
            if token.startswith("▁") or token.startswith("Ġ"):
                if current_word:
                    cleaned_tokens.append(current_word)
                current_word = token[1:]
            else:
                current_word += token
        if current_word:
            cleaned_tokens.append(current_word)
        return cleaned_tokens


    def normalize_pred_label(predicted_label):
        lowered = predicted_label.lower()
        if "yes" in lowered:
            return "yes"
        elif "no" in lowered:
            return "no"
        return predicted_label

    def get_meaningful_pred_label(self, token_id):
        label = self.tokenizer.decode([token_id]).strip()
        if label in ["", "<eos>", "</s>"]:
            return "(No output / End of sentence)"
        return label

    def get_attributions_and_pred(self, input_text):
        """
        Compute Integrated Gradients for the last generated token.
        """
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids

        outputs = self.model.generate(input_ids, max_new_tokens=15)
        generated_token_id = outputs[0, -1].item()
        predicted_label_raw = self.get_meaningful_pred_label(generated_token_id)

        embedding_layer = self.model.get_input_embeddings()
        input_embeddings = embedding_layer(input_ids).to(self.device)
        input_embeddings.requires_grad_()

        def forward_func(embeddings):
            outputs = self.model(inputs_embeds=embeddings)
            last_token_logits = outputs.logits[:, -1, :]
            selected_logits = last_token_logits[:, generated_token_id]
            return selected_logits.unsqueeze(-1)

        ig = IntegratedGradients(forward_func)
        attributions = ig.attribute(input_embeddings, n_steps=50, internal_batch_size=10)
        return attributions, input_ids, predicted_label_raw

    def analyze_text(self, input_text, top_k=3):
        """
        Get the predicted label and top-k tokens with highest attribution scores.
        """
        attributions, input_ids, predicted_label_raw = self.get_attributions_and_pred(input_text)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        attr_np = attributions.detach().cpu().numpy()[0]
        token_importances = np.sum(np.abs(attr_np), axis=1)

        tokens_clean = self.clean_and_reconstruct_tokens(tokens)
        token_importances = token_importances[:len(tokens_clean)]

        simple_label = self.normalize_pred_label(predicted_label_raw)

        sorted_indices = np.argsort(token_importances)[::-1]
        filtered_tokens = [
            (tokens_clean[idx], token_importances[idx])
            for idx in sorted_indices if tokens_clean[idx].strip() != ""
        ]
        top_tokens = filtered_tokens[:top_k]

        print(f"\n🧾 Predicted Label: {simple_label}")
        print(f"🔝 Top {top_k} tokens by attribution score:")
        for token, score in top_tokens:
            print(f"  {token}: {score:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute token-level attributions for text prompts using Integrated Gradients."
    )
    parser.add_argument(
        "--text", type=str, required=True,
        help="The input text or prompt to analyze."
    )
    parser.add_argument(
        "--model", type=str, default="deepseek-ai/deepseek-llm-7b-chat",
        help="Model name or path from Hugging Face."
    )
    parser.add_argument(
        "--top_k", type=int, default=3,
        help="Number of top tokens to display by attribution score."
    )

    args = parser.parse_args()

    analyzer = AttributionAnalyzer(args.model)
    analyzer.analyze_text(args.text, top_k=args.top_k)


if __name__ == "__main__":
    main()
