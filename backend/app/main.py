from fastapi import FastAPI

app=FastAPI(title='SentinelAI')

@app.get('/')
def root():
    return {'status':'ok'}
