# CoinCounter

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Table of Contents

- [About](#about)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [License](#license)

## About
The Coin Counter App is a web application designed to assist users in identifying and counting Euro coins. Utilizing machine learning and image processing techniques, the app can accurately detect various Euro coin denominations from uploaded images. This project was developed as part of a university project by Aref Hasan and Nik Yakovlev in 2024 at The Baden-Württemberg Cooperative State University (DHBW) Mannheim in Germany


## Team Members:
[Nik Yakovlev](https://github.com/nikyak10) 
[Aref Hasan](https://github.com/aref-hasan) 


## Getting Started

### Installation

1. Clone the repository

   ```bash
   git clone aref-hasan/CoinCounter

2. Navigate to the project directory
   ```bash
   cd CoinCounter

3. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt

4. Run the Flask application:
    ```bash
    python app/app.py

5. Access the app:
   open a web browser and go to http://127.0.0.1:5000 to access the Coin Counter App.
   
## Usage 

1. Upload an Image:

On the home page, click on the "Upload an Image" button to upload an image of coins from your device.
Alternatively, you can take a photo using your device's camera.

2. View Results:

After uploading the image, the app will process it and display the original image alongside the image with bounding boxes drawn around detected coins.
The detected coins and their total value will be displayed below the images.

3. Mobile Upload:

Scan the provided QR code to access the upload page on your mobile device.
Upload a photo from your phone, and the app will process the image and display the results on your laptop.


## Contributing
We welcome contributions to enhance the Coin Counter App! If you have any ideas, suggestions, or improvements, feel free to submit a pull request or open an issue.


## License
This project is licensed under the MIT License - see the LICENSE file for details.

