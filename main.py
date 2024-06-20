from openai import AsyncOpenAI
from psycopg2.extras import execute_values
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import tiktoken
import psycopg2
import csv, os, shutil, math
import asyncio, json
from itertools import chain, zip_longest
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
conn = None


@app.get("/")
def health_check():
    return {"status": "success"}


def reconnect():
    global conn
    if conn and conn.closed == 0:
        return
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres.ajfdxbdhabbkupdltrjt",
        password="QkpEcj6XVEf_!P#",
        host="aws-0-ap-southeast-1.pooler.supabase.com",
    )


async def get_embedding(text: str, model="text-embedding-3-small"):
    res = await client.embeddings.create(input=text, model=model)
    return res.data[0].embedding


# Helper func: calculate number of tokens
def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    if not string:
        return 0
    # Returns the number of tokens in a text string
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# Helper function: split documents into chunks
async def split_documents(id, content):
    # list for chunked content and embeddings

    new_list = []
    token_len = num_tokens_from_string(content)
    if token_len <= 512:
        new_list.append([id, content, token_len])
    else:
        # add content to the new list in chunks
        start = 0
        ideal_token_size = 512
        # 1 token ~ 3/4 of a word
        ideal_size = int(ideal_token_size // (4 / 3))
        end = ideal_size
        # split text by spaces into words
        words = content.split()

        # remove empty spaces
        words = [x for x in words if x != " "]

        total_words = len(words)

        # calculate iterations
        chunks = total_words // ideal_size
        if total_words % ideal_size != 0:
            chunks += 1

        new_content = []
        for j in range(chunks):
            if end > total_words:
                end = total_words
            new_content = words[start:end]
            new_content_string = " ".join(new_content)
            new_content_token_len = num_tokens_from_string(new_content_string)
            if new_content_token_len > 0:
                new_list.append([id, new_content_string, new_content_token_len])
            start += ideal_size
            end += ideal_size

    for i in range(len(new_list)):
        text = new_list[i][1]
        embedding = await get_embedding(text)
        new_list[i].append(embedding)

    return new_list


def check_validity(product_id, description):
    # Check if product_id is an integer and not NaN
    if not isinstance(product_id, int) or math.isnan(product_id):
        return False

    # Check if description is a non-empty string
    if not isinstance(description, str) or description.strip() == "":
        return False

    return True


async def insert_embedding(product_id: int, description: str):
    reconnect()

    print(product_id)

    if check_validity(product_id, description):
        print(product_id)
        chunks = await split_documents(product_id, description)
        data_list = [(int(row[0]), row[1], int(row[2]), row[3]) for row in chunks]
        cur = conn.cursor()
        cur.execute("DELETE FROM embeddings WHERE product_id = %s;", (product_id,))
        execute_values(
            cur,
            "INSERT INTO embeddings (product_id, chunk, tokens, embedding) VALUES %s;",
            data_list,
        )
        conn.commit()


class Product(BaseModel):
    id: int
    description: str


@app.post("/update-product/")
async def update_product(product: Product):
    try:
        await insert_embedding(product.id, product.description)
        return {"success": "Embedded a new product successfully."}
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))


@app.post("/bulk-update/")
async def bulk_update(csvfile: UploadFile = File(...)):
    try:
        directory = "./data/csv"
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(f"{directory}/{csvfile.filename}", "wb") as buffer:
            shutil.copyfileobj(csvfile.file, buffer)

        file_path = f"{directory}/{csvfile.filename}"

        try:
            file = open(file_path, "r", encoding="utf-8")
        except UnicodeDecodeError:
            print("UnicodeDecodeError: trying different encoding")
            file = open(file_path, "r", encoding="ISO-8859-1")

        # Create a CSV dictionary reader
        reader = csv.DictReader(file)
        # Lowercase the column names
        reader.fieldnames = [name.lower() for name in reader.fieldnames]

        # Iterate through each row in the CSV
        for row in reader:
            # Access the id and description fields from the row
            # We're using .get() to return None if the keys don't exist
            product_id = row.get("id")
            product_description = row.get("description")

            # Call the setProduct function with obtained id and description
            await insert_embedding(int(product_id), product_description)

        return {"success": "Embedded bulk products successfully."}
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))


res = {}


async def match(product_id: int, chunk: str, keyword: str):
    # response = await client.chat.completions.create(
    #     temperature=0,
    #     model="gpt-3.5-turbo-1106",
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": f"You are 0/1 classifier. If the following chunks doesn't match the search keyword '{keyword}' or has negative meaning from it, output 0. Otherwise, 1. Output 0 or 1 only. No descriptions.\n\n{chunk}",
    #         },
    #     ],
    # )
    # print(response.choices[0].message.content, keyword, product_id, chunk)
    # return product_id if response.choices[0].message.content == "1" else 0
    return product_id


async def enrich(keyword: str):
    response = await client.chat.completions.create(
        temperature=0,
        model="gpt-3.5-turbo-1106",
        messages=[
            {
                "role": "system",
                "content": f"""I'm running a E-commerce company.
The product types are A2 Ghee, Almond Butter, Almonds, Aloe Vera Juice, Amla Juice, Animal Food, Apparels & Toys, Apple Cider Vinegar, Ashwagandha, Bamboo Items, Bamboo Toothbrush, Bamboo Towel, Basil, Bilona Ghee, Black Tea, Body Care, Brass Utensils, Brown Sugar, Candles, Chamomile Tea, Chia Seeds, Chilli & Pepper, Coconut Oil, Coconut Products, Coffee, cold-pressed oil, Copper Bottle, Copper Utensils, Ecofriendly Products, Face Care, Flax Seeds, Ginger Tea, Green Tea, Groundnut oil, Healthy Millet, Herbal Tea, Herbs, Honey, Household & Cleaning, Lemongrass Tea, Makhana, Masala Tea, Mustard oil, Name, Organic Clothing, Baby Food, Best Selling Products, Beverages, Peanut Butter, Personal Care, Rosemary Tea, Scented Candles, Sesame oil, Tea, Toys, Coriander, Cow Ghee, Dry Fruits, Tulsi Tea, Edible Oil, Edible Seeds, Essential Oil, Exclusive Products, Eye Care, Farm Supplies, Fitness & Supplements, Flakes, Flours, Ghee, Giloy Juice, Grocery, Hair Care, Hair Color, Health Drinks, Healthy Breakfast, Hemp, Hemp Products, Herbal Cleaning, Herbal Powder, Home Decor, Household, Iced Tea, Jute Bags, Kitchen Essentials, Kombucha, Lakadong, Lip Care, Matcha Tea, Mint Tea, Moringa, Nut Butter, Nutraceuticals, Oregano, Organic Compost, Organic Pesticides, Organic Seeds, Pickles, Pooja Agarbatti, Pooja Essentials, Pooja Thali, Pumpkin Seeds, Recommended Products, Repellent, Shampoo & Conditioner, Shilajit, shop, Skin Care, Soap Bars, Spices & Masala, Spray, Sugar, Tea Bags, Triphala Powder, Tulsi Powder, Turmeric Tea, Virgin oil, Water Bottle, Wheat Grass Juice, Wooden items, Turmeric, Walnut, Wooden Bottle, Yoga, Yoga Accessories, Yoga Mats, Hair Oil.
If the keyword is about improvement in health related issues, write products or ingredients to improve it.
If the keyword is about disease related issues, write products or ingredients to treat it.
If the keyword is about skin care related issues, write products or ingredients realted to skin treatment it.
If the keyword is about household cleaning related issues, write products or ingredients to clean the household.
If the keyword is about culinary related issues, write products or ingredients to delicious and innovative dishes.
If the keyword is about meditation, excercises related issues, write products or ingredients related to yoga, herbal teas, healthy food that can help alleviate stress and promote inner peace.
If the keyword is about organic farming or natural farming related issues, write products or ingredients that heaps grow organically or naturally.
If the keyword is about gardening related issues, write products or ingredients that are useful in gardening.
If the keyword is about allergies from a partiular product or ingredient related issues, write products or ingredients that can be used as the substitute.
If the keyword is about particular age group, write products or ingredients that can be used by that age group.
If the keyword is about advantages of certain product or ingredient, write advantages of that certain product.
If the keyword is about comparison between two products or ingredients, write the comparison between products or ingredients.
If the keyword is about body pain related issues, write products or ingredients to reduce the pain.
If the keyword is about strengthen hair or strength any body part related issues, write products or ingredients to strengthen the particular area.
If the keyword is about organic clothing related issues, write products or materials about organic clothing.
If the keyword is about hygiene related issues, write products or ingredients to maintain hygiene.
If the keyword is about oral hygiene related issues, write products or ingredients related to oral hygiene.
If the keyword is about loosing weight related issues, write products or ingredients to reduce weight.
If the keyword is about gaining weight related issues, write products or ingredients to gain weight.
If the keyword is about body buidling related issues, write products or ingredients to build body naturally and organically.
If the keyword is about pregnancy related issues, write products or ingredients that can be used during pregnancy.
If the keyword is about healthy diet, write products or ingredients related to healthy diet.
If the keyword is about eco-friendly or sustainable products, write products or ingredients or materials related to eco-friendly or sustainable products.
If the keyword is about digestion related issues, write products or ingredients to improve digestion.
If the keyword is about age related issues, write products or ingredients according to age.
If the keyword is about handmade products, write products or ingredients or materials that are homemade.
If the keyword is about utensils related issues, write products or materials related to utensils.
If the keyword is about immunity related issues, write products or ingredients to boost immunity.
If the keyword is about animals related issues, write products or ingredients related to animals.
If the keyword is about beauty related issues, write products or ingredients to enhance beauty naturally.

Now help me to enrich the following keyword to search products more correctly.

Keyword: {keyword}""",
            },
        ],
    )
    return response.choices[0].message.content


async def get_products(keyword: str, count: str, pre_ids: list):
    embedding = await get_embedding(keyword)
    cur = conn.cursor()
    # Get the top 30 most similar documents using the KNN <=> operator
    cur.execute(
        f"SELECT product_id, chunk FROM embeddings ORDER BY embedding <=> %s LIMIT {count}",
        (str(embedding),),
    )
    products = cur.fetchall()
    ids = list(
        dict.fromkeys(
            await asyncio.gather(
                *[
                    match(product[0], product[1], keyword)
                    for product in products
                    if product[0] not in pre_ids
                ]
            )
        )
    )
    if 0 in ids:
        ids.remove(0)
    return ids


async def pre_search(keyword: str):
    keys = keyword.split(" ")
    keys.insert(0, "")
    keys.append("")
    cur = conn.cursor()
    # Get the top 30 most similar documents using the KNN <=> operator
    cur.execute(
        f"SELECT product_id, chunk FROM embeddings WHERE chunk LIKE %s LIMIT 100",
        ("%".join(keys),),
    )
    products = cur.fetchall()
    ids = list(
        dict.fromkeys(
            await asyncio.gather(
                *[match(product[0], product[1], keyword) for product in products]
            )
        )
    )
    if 0 in ids:
        ids.remove(0)
    return ids


async def search(keyword: str):
    reconnect()
    pre_ids = await pre_search(keyword)
    keyword = await enrich(keyword)
    keys = keyword.split("\n")
    count = (100 - len(pre_ids)) // len(keys)
    products = await asyncio.gather(
        *[get_products(key, count, pre_ids) for key in keys]
    )
    res = list(chain(*zip_longest(*products)))
    res = [i for i in res if i is not None]
    return pre_ids + list(dict.fromkeys(res))


@app.get("/generate-image/")
async def generate_image(query: str, gender: str, age: str, skin_type: str = ""):
    for _ in range(10):
        try:
            response_image = await client.images.generate(
                model="dall-e-3",
                prompt=f"""Generate exactly two images to show comparisons for the query.
                Don't display any letters in the images.
                In every query we need to show the first image related to that health issues related to the query,
                then second image we need to show is the improved helathy body after using the products suggested by our model related to the query.
                e.g.
                If the query is related to comparison, show one image with one unhealthy or sad type and another image with opposite positive type (happy, healthy).
                Like in 'cold pressed or wood pressed oil', show one image of cold pressed and another of wood pressed oil.
                If the query is 'i want to lose weight', the first image is a fat man and the second image is slim man.
                if the query is substitute of wheat, the first image is an unhealthy man using wheat like who is allergic to wheat,
                then the second image is a healthier looking man who has started using substitutes of wheat products.
                if the query is related to having diabetes, the first image is to show a body having diabetes,
                then the second image will be the body that will become after using our products related to the query (healthy body)
                
                query: {query}
                gender: {gender}
                age: {age}"""
                + (f"\nskin type: {skin_type}" if skin_type != "" else ""),
                size="1792x1024",
                quality="standard",
                n=1,
            )

            # Check if there are any text in generated image
            response_text = await client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Is there any words or letters in the image except some symbols like '$' or '?'? Your response should be as short as possible, with only a few key words.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": response_image.data[0].url,
                                },
                            },
                        ],
                    }
                ],
                max_tokens=100,
            )

            # Extract in JSON format
            sample_output = {"exist": True}

            response_bool = await client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {
                        "role": "user",
                        "content": f"""
    From the below setence, return if there was any words or letters in the image in JSON format: {json.dumps(sample_output)}
    -------------------
    {response_text.choices[0].message.content}
    -------------------
    """,
                    }
                ],
                response_format={"type": "json_object"},
            )

            json_result = json.loads(response_bool.choices[0].message.content)

            # if text exist in the image, re-generate image
            if "exist" in json_result and json_result["exist"] == True:
                continue
            return {"image": response_image.data[0].url}

        except Exception as err:
            continue
        #  raise HTTPException(status_code=400, detail=str(err))
    return {"image": "failed"}


@app.get("/search-product/")
async def search_product(keyword: str):
    try:
        return await search(keyword)
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))


@app.post("/search-by-voice/")
async def search_by_voice(audiofile: UploadFile = File(...)):
    try:
        directory = "./data/audio"
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(f"{directory}/{audiofile.filename}", "wb") as buffer:
            shutil.copyfileobj(audiofile.file, buffer)

        file = open(f"{directory}/{audiofile.filename}", "rb")
        transcript = await client.audio.transcriptions.create(
            model="whisper-1", file=file
        )
        print(transcript.text)
        return await search(transcript.text)
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))


# Endpoint to get credentials by website
@app.get("/credentials/{website}")
def get_credentials(website: str):
    try:
        reconnect()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT url, username, password FROM emails WHERE username NOT LIKE 'Password:%%' AND url LIKE %s",
            (f"%{website}%",),
        )
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        if not result:
            raise HTTPException(status_code=404, detail="Credentials not found")
        return result
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)
