# ai-planet-industry-usecase-genreator-copilot

<img width="400" src="https://framerusercontent.com/images/pFpeWgK03UT38AQl5d988Epcsc.svg"> 


# Industry Use Case Generator

This project generates industry research, AI use cases, and datasets based on a company name. The generated reports can be downloaded as PDFs, and the application features a chat interface for asking questions based on the generated reports.

![Screenshot 2025-03-22 210859](https://github.com/user-attachments/assets/197c07e5-34b3-440d-9022-bec682d658fc)
![Screenshot 2025-03-22 211041](https://github.com/user-attachments/assets/e9571960-35e2-4149-96a1-32c1e1e99473)


<img width="300" src="https://github.com/user-attachments/assets/a90c49c6-5373-4045-86bc-46a25f420502"> 

## Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/Ganeshkharde1/industry-usecase-genreator-copilot.git

```
### Navigate to the Project Directory

```bash
cd industry-usecase-genreator-copilot
```
Install Required Dependencies
```bash
pip install -r requirements.txt
```
### Update API Keys

Create a .env file in the project root directory.
Add your API keys inside the .env file:

```bash
TAVILY_API_KEY=your_tavily_api_key
GROQ_API_KEY=your_groq_api_key
```
### Run the Streamlit App
```bash
python -m streamlit run app.py
```
### Usage
1. Enter the company name in the input field.
2. Click on "Generate Report" to fetch industry research, AI use cases, and datasets.
3. Download the generated Industry Report, AI Use Cases, and AI/ML Datasets as PDFs.
4. Use the chat interface on the sidebar to ask questions based on the generated reports.
   
## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
