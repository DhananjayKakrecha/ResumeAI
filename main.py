from flask import Flask,request,render_template,session,redirect,url_for
import os
import pickle
import PyPDF2
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
from pdf2image import convert_from_path
import pytesseract
import pdfplumber
import mysql.connector as mys

rf_classifier_categorization = pickle.load(open("models/rf_classifier_categorization.pkl",'rb'))
tfidf_vectorizer_categorization = pickle.load(open("models/tfidf_vectorizer_categorization.pkl",'rb'))
rf_classifier_job_recommendation = pickle.load(open("models/rf_classifier_job_recommendation.pkl",'rb'))
tfidf_vectorizer_job_recommendation = pickle.load(open("models/tfidf_vectorizer_job_recommendation.pkl",'rb'))

load_dotenv()

genai.configure(api_key="AIzaSyDjYT39ug61cSBv2QV3ocGn3IkBb5meJWU")


def extract_text_from_pdf2(pdf_path):
    text = ""
    try:
        # Try direct text extraction
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

        if text.strip():
            return text.strip()
    except Exception as e:
        print(f"Direct text extraction failed: {e}")

    # Fallback to OCR for image-based PDFs
    print("Falling back to OCR for image-based PDF.")
    try:
        images = convert_from_path(pdf_path)
        for image in images:
            page_text = pytesseract.image_to_string(image)
            text += page_text + "\n"
    except Exception as e:
        print(f"OCR failed: {e}")

    return text.strip()


# Function to get response from Gemini AI
def analyze_resume(resume_text, job_description=None):
    if not resume_text:
        return {"error": "Resume text is required for analysis."}

    model = genai.GenerativeModel("gemini-1.5-flash")

    base_prompt = f"""
    You are an experienced HR with Technical Experience in the field of any one job role from Data Science, Data Analyst, DevOPS, Machine Learning Engineer, Prompt Engineer, AI Engineer, Full Stack Web Development, Big Data Engineering, Marketing Analyst, Human Resource Manager, Software Developer your task is to review the provided resume.
    Please share your professional evaluation on whether the candidate's profile aligns with the role.ALso mention Skills he already have and suggest some skills to improve his resume , also suggest some course he might take to improve the skills.Highlight the strengths and weaknesses.

    Resume:
    {resume_text}
    """

    if job_description:
        base_prompt += f"""
        Additionally, compare this resume to the following job description:

        Job Description:
        {job_description}

        Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
        """

    response = model.generate_content(base_prompt)

    analysis = response.text.strip()
    return analysis

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def predict_category(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
    predicted_category = rf_classifier_categorization.predict(resume_tfidf)[0]
    return predicted_category

def job_recommendation(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
    predicted_category = rf_classifier_job_recommendation.predict(resume_tfidf)[0]
    return predicted_category

def extract_contact_number_from_resume(text):
    contact_number = None
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    if match:
        print("match found")
        contact_number = match.group()
    return contact_number

def extract_email_from_resume(text):
    email = None

    # Use regex pattern to find a potential email address
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    if match:
        email = match.group()

    return email

def extract_skills_from_resume(text):
    skills = []

    # List of predefined skills
    skills_list = [
        'Python', 'Data Analysis', 'Machine Learning', 'Communication', 'Project Management', 'Deep Learning', 'SQL',
        'Tableau',
        'Java', 'C++', 'JavaScript', 'HTML', 'CSS', 'React', 'Angular', 'Node.js', 'MongoDB', 'Express.js', 'Git',
        'Research', 'Statistics', 'Quantitative Analysis', 'Qualitative Analysis', 'SPSS', 'R', 'Data Visualization',
        'Matplotlib',
        'Seaborn', 'Plotly', 'Pandas', 'Numpy', 'Scikit-learn', 'TensorFlow', 'Keras', 'PyTorch', 'NLTK', 'Text Mining',
        'Natural Language Processing', 'Computer Vision', 'Image Processing', 'OCR', 'Speech Recognition',
        'Recommendation Systems',
        'Collaborative Filtering', 'Content-Based Filtering', 'Reinforcement Learning', 'Neural Networks',
        'Convolutional Neural Networks',
        'Recurrent Neural Networks', 'Generative Adversarial Networks', 'XGBoost', 'Random Forest', 'Decision Trees',
        'Support Vector Machines',
        'Linear Regression', 'Logistic Regression', 'K-Means Clustering', 'Hierarchical Clustering', 'DBSCAN',
        'Association Rule Learning',
        'Apache Hadoop', 'Apache Spark', 'MapReduce', 'Hive', 'HBase', 'Apache Kafka', 'Data Warehousing', 'ETL',
        'Big Data Analytics',
        'Cloud Computing', 'Amazon Web Services (AWS)', 'Microsoft Azure', 'Google Cloud Platform (GCP)', 'Docker',
        'Kubernetes', 'Linux',
        'Shell Scripting', 'Cybersecurity', 'Network Security', 'Penetration Testing', 'Firewalls', 'Encryption',
        'Malware Analysis',
        'Digital Forensics', 'CI/CD', 'DevOps', 'Agile Methodology', 'Scrum', 'Kanban', 'Continuous Integration',
        'Continuous Deployment',
        'Software Development', 'Web Development', 'Mobile Development', 'Backend Development', 'Frontend Development',
        'Full-Stack Development',
        'UI/UX Design', 'Responsive Design', 'Wireframing', 'Prototyping', 'User Testing', 'Adobe Creative Suite',
        'Photoshop', 'Illustrator',
        'InDesign', 'Figma', 'Sketch', 'Zeplin', 'InVision', 'Product Management', 'Market Research',
        'Customer Development', 'Lean Startup',
        'Business Development', 'Sales', 'Marketing', 'Content Marketing', 'Social Media Marketing', 'Email Marketing',
        'SEO', 'SEM', 'PPC',
        'Google Analytics', 'Facebook Ads', 'LinkedIn Ads', 'Lead Generation', 'Customer Relationship Management (CRM)',
        'Salesforce',
        'HubSpot', 'Zendesk', 'Intercom', 'Customer Support', 'Technical Support', 'Troubleshooting',
        'Ticketing Systems', 'ServiceNow',
        'ITIL', 'Quality Assurance', 'Manual Testing', 'Automated Testing', 'Selenium', 'JUnit', 'Load Testing',
        'Performance Testing',
        'Regression Testing', 'Black Box Testing', 'White Box Testing', 'API Testing', 'Mobile Testing',
        'Usability Testing', 'Accessibility Testing',
        'Cross-Browser Testing', 'Agile Testing', 'User Acceptance Testing', 'Software Documentation',
        'Technical Writing', 'Copywriting',
        'Editing', 'Proofreading', 'Content Management Systems (CMS)', 'WordPress', 'Joomla', 'Drupal', 'Magento',
        'Shopify', 'E-commerce',
        'Payment Gateways', 'Inventory Management', 'Supply Chain Management', 'Logistics', 'Procurement',
        'ERP Systems', 'SAP', 'Oracle',
        'Microsoft Dynamics', 'Tableau', 'Power BI', 'QlikView', 'Looker', 'Data Warehousing', 'ETL',
        'Data Engineering', 'Data Governance',
        'Data Quality', 'Master Data Management', 'Predictive Analytics', 'Prescriptive Analytics',
        'Descriptive Analytics', 'Business Intelligence',
        'Dashboarding', 'Reporting', 'Data Mining', 'Web Scraping', 'API Integration', 'RESTful APIs', 'GraphQL',
        'SOAP', 'Microservices',
        'Serverless Architecture', 'Lambda Functions', 'Event-Driven Architecture', 'Message Queues', 'GraphQL',
        'Socket.io', 'WebSockets'
                     'Ruby', 'Ruby on Rails', 'PHP', 'Symfony', 'Laravel', 'CakePHP', 'Zend Framework', 'ASP.NET', 'C#',
        'VB.NET', 'ASP.NET MVC', 'Entity Framework',
        'Spring', 'Hibernate', 'Struts', 'Kotlin', 'Swift', 'Objective-C', 'iOS Development', 'Android Development',
        'Flutter', 'React Native', 'Ionic',
        'Mobile UI/UX Design', 'Material Design', 'SwiftUI', 'RxJava', 'RxSwift', 'Django', 'Flask', 'FastAPI',
        'Falcon', 'Tornado', 'WebSockets',
        'GraphQL', 'RESTful Web Services', 'SOAP', 'Microservices Architecture', 'Serverless Computing', 'AWS Lambda',
        'Google Cloud Functions',
        'Azure Functions', 'Server Administration', 'System Administration', 'Network Administration',
        'Database Administration', 'MySQL', 'PostgreSQL',
        'SQLite', 'Microsoft SQL Server', 'Oracle Database', 'NoSQL', 'MongoDB', 'Cassandra', 'Redis', 'Elasticsearch',
        'Firebase', 'Google Analytics',
        'Google Tag Manager', 'Adobe Analytics', 'Marketing Automation', 'Customer Data Platforms', 'Segment',
        'Salesforce Marketing Cloud', 'HubSpot CRM',
        'Zapier', 'IFTTT', 'Workflow Automation', 'Robotic Process Automation (RPA)', 'UI Automation',
        'Natural Language Generation (NLG)',
        'Virtual Reality (VR)', 'Augmented Reality (AR)', 'Mixed Reality (MR)', 'Unity', 'Unreal Engine', '3D Modeling',
        'Animation', 'Motion Graphics',
        'Game Design', 'Game Development', 'Level Design', 'Unity3D', 'Unreal Engine 4', 'Blender', 'Maya',
        'Adobe After Effects', 'Adobe Premiere Pro',
        'Final Cut Pro', 'Video Editing', 'Audio Editing', 'Sound Design', 'Music Production', 'Digital Marketing',
        'Content Strategy', 'Conversion Rate Optimization (CRO)',
        'A/B Testing', 'Customer Experience (CX)', 'User Experience (UX)', 'User Interface (UI)', 'Persona Development',
        'User Journey Mapping', 'Information Architecture (IA)',
        'Wireframing', 'Prototyping', 'Usability Testing', 'Accessibility Compliance', 'Internationalization (I18n)',
        'Localization (L10n)', 'Voice User Interface (VUI)',
        'Chatbots', 'Natural Language Understanding (NLU)', 'Speech Synthesis', 'Emotion Detection',
        'Sentiment Analysis', 'Image Recognition', 'Object Detection',
        'Facial Recognition', 'Gesture Recognition', 'Document Recognition', 'Fraud Detection',
        'Cyber Threat Intelligence', 'Security Information and Event Management (SIEM)',
        'Vulnerability Assessment', 'Incident Response', 'Forensic Analysis', 'Security Operations Center (SOC)',
        'Identity and Access Management (IAM)', 'Single Sign-On (SSO)',
        'Multi-Factor Authentication (MFA)', 'Blockchain', 'Cryptocurrency', 'Decentralized Finance (DeFi)',
        'Smart Contracts', 'Web3', 'Non-Fungible Tokens (NFTs)']

    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            skills.append(skill)

    return skills

def extract_education_from_resume(text):
    education = []

    # List of education keywords to match against
    education_keywords = [
        'Computer Science', 'Information Technology', 'Software Engineering', 'Electrical Engineering', 'Mechanical Engineering', 'Civil Engineering',
        'Chemical Engineering', 'Biomedical Engineering', 'Aerospace Engineering', 'Nuclear Engineering', 'Industrial Engineering', 'Systems Engineering',
        'Environmental Engineering', 'Petroleum Engineering', 'Geological Engineering', 'Marine Engineering', 'Robotics Engineering', 'Biotechnology',
        'Biochemistry', 'Microbiology', 'Genetics', 'Molecular Biology', 'Bioinformatics', 'Neuroscience', 'Biophysics', 'Biostatistics', 'Pharmacology',
        'Physiology', 'Anatomy', 'Pathology', 'Immunology', 'Epidemiology', 'Public Health', 'Health Administration', 'Nursing', 'Medicine', 'Dentistry',
        'Pharmacy', 'Veterinary Medicine', 'Medical Technology', 'Radiography', 'Physical Therapy', 'Occupational Therapy', 'Speech Therapy', 'Nutrition',
        'Sports Science', 'Kinesiology', 'Exercise Physiology', 'Sports Medicine', 'Rehabilitation Science', 'Psychology', 'Counseling', 'Social Work',
        'Sociology', 'Anthropology', 'Criminal Justice', 'Political Science', 'International Relations', 'Economics', 'Finance', 'Accounting', 'Business Administration',
        'Management', 'Marketing', 'Entrepreneurship', 'Hospitality Management', 'Tourism Management', 'Supply Chain Management', 'Logistics Management',
        'Operations Management', 'Human Resource Management', 'Organizational Behavior', 'Project Management', 'Quality Management', 'Risk Management',
        'Strategic Management', 'Public Administration', 'Urban Planning', 'Architecture', 'Interior Design', 'Landscape Architecture', 'Fine Arts',
        'Visual Arts', 'Graphic Design', 'Fashion Design', 'Industrial Design', 'Product Design', 'Animation', 'Film Studies', 'Media Studies',
        'Communication Studies', 'Journalism', 'Broadcasting', 'Creative Writing', 'English Literature', 'Linguistics', 'Translation Studies',
        'Foreign Languages', 'Modern Languages', 'Classical Studies', 'History', 'Archaeology', 'Philosophy', 'Theology', 'Religious Studies',
        'Ethics', 'Education', 'Early Childhood Education', 'Elementary Education', 'Secondary Education', 'Special Education', 'Higher Education',
        'Adult Education', 'Distance Education', 'Online Education', 'Instructional Design', 'Curriculum Development'
        'Library Science', 'Information Science', 'Computer Engineering', 'Software Development', 'Cybersecurity', 'Information Security',
        'Network Engineering', 'Data Science', 'Data Analytics', 'Business Analytics', 'Operations Research', 'Decision Sciences',
        'Human-Computer Interaction', 'User Experience Design', 'User Interface Design', 'Digital Marketing', 'Content Strategy',
        'Brand Management', 'Public Relations', 'Corporate Communications', 'Media Production', 'Digital Media', 'Web Development',
        'Mobile App Development', 'Game Development', 'Virtual Reality', 'Augmented Reality', 'Blockchain Technology', 'Cryptocurrency',
        'Digital Forensics', 'Forensic Science', 'Criminalistics', 'Crime Scene Investigation', 'Emergency Management', 'Fire Science',
        'Environmental Science', 'Climate Science', 'Meteorology', 'Geography', 'Geomatics', 'Remote Sensing', 'Geoinformatics',
        'Cartography', 'GIS (Geographic Information Systems)', 'Environmental Management', 'Sustainability Studies', 'Renewable Energy',
        'Green Technology', 'Ecology', 'Conservation Biology', 'Wildlife Biology', 'Zoology']

    for keyword in education_keywords:
        pattern = r"(?i)\b{}\b".format(re.escape(keyword))
        match = re.search(pattern, text)
        if match:
            education.append(match.group())

    return education

def extract_name_from_resume(text):
    name = None

    # Use regex pattern to find a potential name
    pattern = r"(\b[A-Z][a-z]+\b)\s(\b[A-Z][a-z]+\b)"
    match = re.search(pattern, text)
    if match:
        name = match.group()

    return name


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['SECRET_KEY'] = 'resumeai987'

#helper functions
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path, 'r' ,encoding='utf-8') as file:
        return file.read()

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)

    if file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)

    if file_path.endswith(".txt"):
        return extract_text_from_txt(file_path)

    return ""



@app.route("/")
def matchresume():
    return render_template("home.html")

@app.route("/matcher", methods=['GET','POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form.get('job_description')
        resume_files = request.files.getlist('resumes')
        topvalues = int(request.form.get('topValues'))

        resumes = []
        for resume_file in resume_files:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(filename)
            resumes.append(extract_text(filename))

        if not resumes and not job_description:
            return render_template('matchresume.html',message="Please Upload resumes and post job..")

        vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
        #print(vectorizer)
        vectors = vectorizer.toarray()
        job_vector = vectors[0]
        resume_vectors = vectors[1:]
        # print(job_vector)
        # print("======================")
        # print(resume_vectors)
        similarities = cosine_similarity([job_vector],resume_vectors)[0]
        # print(job_vector)
        # print("================")
        # print(resume_vectors)
        # print("================")
        # print(similarities)

        top_indices = similarities.argsort()[-topvalues:][::-1]
        top_resumes = [resume_files[i].filename for i in top_indices]
        similarity_score = [round(similarities[i],2) for i in top_indices]

        return render_template('matchresume.html',message="Top Matching resumes:",top_resumes=top_resumes,similarity_scores=similarity_score)
    return render_template('matchresume.html')

@app.route("/details", methods=['GET','POST'])
def details():
    resume_name = request.form.get('resume')
    filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_name)
    text = extract_text(filename)

    predicted_category = predict_category(text)
    recommended_job = job_recommendation(text)
    phone = extract_contact_number_from_resume(text)
    email = extract_email_from_resume(text)
    extracted_skills = extract_skills_from_resume(text)
    extracted_education = extract_education_from_resume(text)
    name = extract_name_from_resume(text)

    return render_template("resume.html",predicted_category=predicted_category,phone=phone,email=email,name=name,extracted_skills=extracted_skills,extracted_education=extracted_education)


@app.route("/analyze", methods= ['GET','POST'])
def analyze():
    if request.method == 'POST':
        job_description = request.form.get('job_description')
        resume_files = request.files.getlist('resumes')

        resumes = ""
        for resume_file in resume_files:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(filename)

            resumes = extract_text_from_pdf2(filename)

        analysis = analyze_resume(resumes, job_description)

        return render_template("user.html",text=analysis)


#DB Connections
def db_connect():
    conn = mys.connect(host='localhost', user='root', password='root', database='resumeai')
    return conn

def registerUser(username,password,name,email):
    conn = db_connect()
    cursor = conn.cursor()
    sql = "INSERT INTO users (username,password,name,email) VALUES (%s, %s, %s, %s)"
    val = (username,password,name,email)
    cursor.execute(sql,val)
    conn.commit()
    cursor.close()
    conn.close()

def registerHR(username,password,name,email,department):
    conn = db_connect()
    cursor = conn.cursor()
    sql = "INSERT INTO HR (username,password,name,email,department) VALUES (%s, %s, %s, %s, %s)"
    val = (username,password,name,email,department)
    cursor.execute(sql,val)
    conn.commit()
    cursor.close()
    conn.close()

def loginUser(username,password):
    conn = db_connect()
    cursor = conn.cursor()
    sql = "Select * from users where username = %s and password = %s"
    cursor.execute(sql, (username,password))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user

def loginHR(username,password):
    conn = db_connect()
    cursor = conn.cursor()
    sql = "Select * from HR where username = %s and password = %s"
    cursor.execute(sql, (username,password))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user

def loginAd(username,password):
    conn = db_connect()
    cursor = conn.cursor()
    sql = "Select * from Admins where username = %s and password = %s"
    cursor.execute(sql, (username,password))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user

@app.route('/signup')
def signup():
    return render_template("signup.html")

@app.route('/hrsignup')
def signuphr():
    return render_template("hrsignup.html")

@app.route('/register', methods=["GET","POST"])
def register():
    username = request.form["username"]
    password = request.form["password"]
    name = request.form["name"]
    email = request.form["email"]
    registerUser(username,password,name,email)
    print("Registration Done")
    return render_template("login.html")

@app.route('/registerHR', methods=["GET","POST"])
def registerH():
    username = request.form["username"]
    password = request.form["password"]
    name = request.form["name"]
    email = request.form["email"]
    department = request.form['department']
    registerHR(username, password, name, email, department)
    print("Registration Done")
    return render_template("hrlogin.html")

@app.route('/login',methods=["GET"])
def log():
    return render_template("login.html")

@app.route('/hrlogin',methods=["GET"])
def loghr():
    return render_template("hrlogin.html")

@app.route('/adlogin',methods=["GET"])
def logadm():
    return render_template("adminlogin.html")


@app.route('/login', methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]
    user = loginUser(username,password)
    if user:
        session["username"] = username
        return render_template("user.html",
                               username=username)
    else:
        return render_template("error.html")

@app.route('/loginHR', methods=["GET","POST"])
def loginH():
    username = request.form["username"]
    password = request.form["password"]
    user = loginHR(username,password)
    if user:
        session["username"] = username
        return render_template("matchresume.html",
                               username=username)
    else:
        return render_template("error.html")

#Admin
@app.route('/logadmin',methods=["GET","POST"])
def loginAdmin():
    username = request.form["username"]
    password = request.form["password"]
    user = loginAd(username,password)
    if user:
        session["username"] = username
        conn = db_connect()
        cursor = conn.cursor(dictionary=True)
        # Fetch HR and Users
        cursor.execute("SELECT id, name, email, username, password, department FROM hr")
        hr_list = cursor.fetchall()
        cursor.execute("SELECT id, name, email, username, password FROM users")
        user_list = cursor.fetchall()
        cursor.close()
        conn.close()
        return render_template('admin_home.html', hr_list=hr_list, user_list=user_list)
    else:
        return render_template("error.html")

@app.route('/admin',methods=["GET"])
def admin_home():
    conn = db_connect()
    cursor = conn.cursor(dictionary=True)
    # Fetch HR and Users
    cursor.execute("SELECT id, name, email, username, password, department FROM hr")
    hr_list = cursor.fetchall()
    cursor.execute("SELECT id, name, email, username, password FROM users")
    user_list = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('admin_home.html', hr_list=hr_list, user_list=user_list)

@app.route('/delete_user', methods=['POST'])
def delete_user():
    user_id = request.form.get('user_id')
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
    conn.commit()
    cursor.close()
    conn.close()
    return redirect(url_for('admin_home'))

@app.route('/delete_hr', methods=['POST'])
def delete_hr():
    hr_id = request.form.get('hr_id')
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM hr WHERE id = %s", (hr_id,))
    conn.commit()
    cursor.close()
    conn.close()
    return redirect(url_for('admin_home'))

@app.route('/logout')
def logout():
    return render_template("home.html")

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)