FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . ./

# Install the required dependencies
RUN pip install -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run app.py when the container launches
# CMD python -m streamlit run final.py
CMD streamlit run final.py --server.port 8080 --server.address 0.0.0.0 --browser.serverAddress 0.0.0.0


