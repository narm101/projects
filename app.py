import os

import replicate
import streamlit as st
from dotenv import load_dotenv
from elevenlabs import generate
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
eleven_api_key = os.getenv("ELEVEN_API_KEY")

llm = OpenAI(temperature=.5)

def generate_recipe(food, calories):
    
    #Creating the prompt
    prompt = PromptTemplate(
        input_variables=("food","calories"),
        template=""""

        You are an experienced chef, please generate a recipe for the {food}
        that has a maximum of {calories} calories.
         """
    
    )
    #creating a chain on langchain takes in the prompt and the open AI model
    llm_chain = LLMChain(
    llm=llm,
    prompt=prompt)
    
    #this generates the output of the prompt
    recipe = llm_chain.run({

        "food": food,
        "calories": calories})
    
    return recipe

    
def generate_audio(recipe,voice):
    audio = generate(text=recipe, voice=voice,api_key=eleven_api_key)

    return audio

def generate_image(food):
    output = replicate.run(
        "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
     input={"prompt": food}
)
    return output

def app():
    st.title("Recipe Generator")
    
    with st.form(key="my_form"):
        food = st.text_input("What food do you want to cook?")
        calories = st.number_input("How many calories do you want to eat?")

        voice_options = ["Bella","Domi","Rachel","Antoni","Elli",'Josh']
        voice = st.selectbox("Which voice do you want to use?",voice_options)

        submit_button = st.form_submit_button(label="Generate Recipe")

    if submit_button:
        print('button was clicked')

        recipe =  generate_recipe(food, calories)
       
        st.markdown(recipe)

        st.audio(generate_audio(recipe,voice))
        


        images = generate_image(food)
        st.image(images[0])



   

if __name__ == "__main__":
    app()