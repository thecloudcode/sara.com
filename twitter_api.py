import pandas as pd
from fastapi import FastAPI
from itertools import cycle
import uvicorn

app = FastAPI()

data1 = pd.read_csv("./Data/tweets.csv")[['location','text']]
data2 = pd.read_csv("./Data/tweets2.csv")[['location','text']]
data = pd.concat([data1,data2])
data['location'] = data['location'].fillna("Not Specified")
data_cycle = cycle(data.to_dict(orient='records'))
print(data.shape)
@app.get("/tweet")
async def get_tweets():
    try:
        tweets = next(data_cycle)
        return tweets
    except StopIteration:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)