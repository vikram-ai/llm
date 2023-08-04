import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False
    
class llm_model:
    def __init__(self, model_name = "stabilityai/stablelm-tuned-alpha-3b") -> None:
        torch_dtype = "float16"
        load_in_8bit = False
        self.model_name = model_name
        device_map = "auto"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, torch_dtype),
            load_in_8bit=load_in_8bit,
            device_map=device_map,
            offload_folder="./offload",
        )
    
    def generate(self, user_prompt):
        if "tuned" in self.model_name:
            # Add system prompt for chat tuned models
            system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
            - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
            - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
            - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
            - StableLM will refuse to participate in anything that could harm a human.
            """
            prompt = f"{system_prompt}<|USER|>{user_prompt}<|ASSISTANT|>"
        else:
            prompt = user_prompt
        max_new_tokens = 1000
        temperature = 0.7
        top_k = 0
        top_p = 0.9
        do_sample = True
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs.to(self.model.device)
        tokens = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()])
        )

        # Extract out only the completion tokens
        completion_tokens = tokens[0][inputs['input_ids'].size(1):]
        completion = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)
        return completion

