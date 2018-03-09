docker build -t "attngan" .
REM docker run -it -e BLOB_KEY=KEY -p 5678:8080 attngan
REM curl -H "Content-Type: application/json" -X POST -d '{"caption":"the bird has a yellow crown and a black eyering that is round"}' http://localhost:5678/attgan/api/v1.0/bird