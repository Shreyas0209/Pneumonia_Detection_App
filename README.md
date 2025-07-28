# Pneumonia_Detection_App

Author: Shreyas J
Department: Computer Science and Design, 4th Year, Engineering
Institution: S J C Institute of Technology
Contact: shreyasj6675@gmail.com

🩺 Project Overview
This project is a professional-grade web application for automated pneumonia detection from chest X-ray images using deep learning. It leverages a fine-tuned InceptionV3 model to deliver high accuracy (94%) and features an intuitive, risk-tiered (green/yellow/red) interface so users receive actionable clinical feedback instantly.

You can upload a chest X-ray (JPEG/PNG), and the app returns the predicted status—"NORMAL" or "PNEUMONIA"—with a confidence score and color-coded risk assessment.

📁 Repository Structure
chestxray-app/

file 1 : app.py            # Main Flask application

file 2 : requirements.txt  # Project dependencies

file 3 : README.md         # Project documentation (this file)

Folder 1 : model
i) Filer 1.1 : best_inceptionv3_model.keras  # Trained Keras model file

Folder 2 : static

I) Folder 2.1 : css
i) File 2.1.1 : style.css  # Custom styles.

II) Folder 2.2 : images 
i) image : favicon.png  # (optional) Project icon.

III) Folder 2.3 : uploads
i)  # Uploaded X-rays (auto-created if not present).

Folder 3 : templates

i) File 3.1 : index.html    # Web interface template.
ii) File 3.2 : result.html

Folder 4 : utils

i) File 4.1 : prepocessing.py  # Image preprocessing function


In order to obtain the model, You need to run the model1.py program.
Store the trained model in the Model folder.

🖼️ Usage
Navigate to the homepage: See a clean web interface with upload functionality.

Drag and drop or select a chest X-ray image (JPEG or PNG, ≤5MB recommended).

Click "Analyze".

View your prediction:

Label: "NORMAL" or "PNEUMONIA"

Confidence: e.g., 96.25%

Risk Tier: Green, Yellow, or Red color-coded bar and message

All image processing is local—no uploads or data storage beyond your browser session.

🟢🔶🔴 Risk Tier System
Tier	Probability Range	Color	Clinical Meaning
Green	≤ 0.40	Green	Safe: Unlikely pneumonia
Yellow	0.41–0.80	Yellow	Caution: Possible pneumonia, consider consultation
Red	> 0.80	Red	High risk: Immediate medical attention advised
🎨 Customization
To update styling: Modify static/css/styles.css

To edit front-end or add pages: Change templates/index.html

To extend model pre-processing: Update utils/preprocessing.py

🌐 Deployment
The app runs locally by default.

Deploy for free on Render, PythonAnywhere, or Replit Cloud, or on your own Linux server.

For public deployment, ensure debug=False and set up HTTPS.

📑 Citation
If you use this project for academic or demonstration purposes, please acknowledge:

Shreyas J, Department of Computer Science and Design, S J C Institute of Technology.

🤝 Contributing & Issues
Contributions: PRs and suggestions are welcome!

For bugs/help: Please open an issue or email shreyasj6675@gmail.com

🔒 Disclaimer
This tool is intended for educational and research purposes only.

It is NOT a substitute for professional medical advice.

Always consult a qualified healthcare provider for medical interpretation or decisions.

📚 References
Kaggle Chest X-Ray Pneumonia Dataset

Rajpurkar, P. et al. "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning." arXiv:1711.05225

Szegedy, C., et al. "Rethinking the Inception Architecture for Computer Vision." CVPR 2016

Enjoy using and sharing your chest X-ray pneumonia detection AI!
For any issues, please reach out via email or GitHub.
