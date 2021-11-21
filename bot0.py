#importy
import random
import discord
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import tensorflow

#deklarace dulezitych veci
vocab_size = 300000
model = tensorflow.keras.models.load_model("model6.h5")
df = pd.read_csv("training data/train_8.A.csv")
tokenizer = Tokenizer(oov_token="<OOV>",num_words=vocab_size)
tokenizer.fit_on_texts(df["msg"].astype(str))


#instance tridy Client
bot = discord.Client()

@bot.event
async def on_ready():
    print("ready")

#kdyz nekdo posle zpravu
@bot.event
async def on_message(message):
    if message.content.startswith(";"):
        try:
            if message.author.id != 832961508802560020:
                async with message.channel.typing():

                    seq = tokenizer.texts_to_sequences([message.content])
                    padded = pad_sequences(seq,maxlen= 100)

                    prdct = np.argmax(model.predict(padded))


                    await message.channel.send(df["reaction"][prdct])
        except Exception as e:
            print(e)
            await message.channel.send(f"{str(random.choice(df['msg'].values))}")


bot.run(input("Token: "))