#pip install transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#pretrained model
model_name = "Vamsi/T5_Paraphrase_Paws"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


#Sentences paraphrasing 
def get_paraphrased_sentences(model, tokenizer, sentence, num_return_sequences=5, num_beams=5,max_length=20):

  # tokenize the text to be form of a list of token IDs
  inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")

  # generate the paraphrased sentences
  outputs = model.generate(
    **inputs,
    num_beams=num_beams,
    num_return_sequences=num_return_sequences,
  )
  # decode the generated sentences using the tokenizer to get them back to text
  return tokenizer.batch_decode(outputs, skip_special_tokens=True)

sentence = "Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences."
outputs = get_paraphrased_sentences(model, tokenizer, sentence, num_beams=5, num_return_sequences=5, max_length=20)
print(outputs)