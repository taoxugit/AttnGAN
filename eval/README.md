# AttnGAN Eval API
Model evaluation code is extracted here in order to create a separate inference mode for the project. The evaluation code is then embedded into a flask app that accepts API requests. 
There are two docker files:
1. [dockerfile.cpu](dockerfile.cpu) - creates a CPU bonud container
2. [dockerfile.gpu](dockerfile.gpu) - creates a GPU bound container

# Requirements
The app uses Azure Blob Storage as an image repository as well as Application Insights for logging telemetry.

# Running the Application
There is a three step process running the application and generating bird images.
1. Create the container (optionally choose the cpu or gpu dockerfile: 
   ```
   docker build -t "attngan" -f dockerfile.cpu .
   ``` 
2. Run the container (replace the key's with the appropriate blob storage location as well as App Insights Key): 
    ```
    docker run -it -e BLOB_KEY=KEY -e TELEMETRY=TELEMETRY_KEY -p 5678:8080 attngan
    ```
3. Call the API: 
   ```
   curl -H "Content-Type: application/json" -X POST -d '{"caption":"the bird has a yellow crown and a black eyering that is round"}' http://locahost:5678/api/v1.0/bird
   ```

# Images
You should have your very own image generator.


