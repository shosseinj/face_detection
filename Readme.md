

## Running Qdrant

```
cd Embedding/
docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant 
```


```
cd Detection
python main.py --collection n3 --threshold 0.01 --webCam False
```