from peft import PeftConfig, PeftModel

config = PeftConfig.from_pretrained('./finetuned_falcon')
finetuned_model = PeftModel.from_pretrained(falcon_model, './finetuned_falcon')


text4 = "A 25-year-old female presents with swelling, pain, and inability to bear weight on her left ankle following a fall during a basketball game where she landed awkwardly on her foot. The pain is on the outer side of her ankle. What is the likely diagnosis and next steps?"

inputs = tokenizer(text4, return_tensors="pt").to("cuda:0")
outputs = finetuned_model.generate(input_ids=inputs.input_ids, max_new_tokens=75)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


