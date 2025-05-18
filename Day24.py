import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class ResumeMatchingSystem:
    """Advanced resume matching system with text preprocessing, 
    visualization, and skill-based matching."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Common technical skills dictionary with categories
        self.skills_dict = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'r'],
            'data_science': ['machine learning', 'deep learning', 'neural networks', 'data mining', 'statistics',
                            'nlp', 'natural language processing', 'computer vision', 'predictive modeling'],
            'data_engineering': ['sql', 'mysql', 'postgresql', 'mongodb', 'database', 'data warehouse', 
                               'etl', 'hadoop', 'spark', 'kafka', 'airflow'],
            'cloud': ['aws', 'azure', 'gcp', 'google cloud', 'cloud computing', 'docker', 'kubernetes', 'serverless'],
            'tools': ['git', 'github', 'jira', 'jenkins', 'tableau', 'power bi', 'excel', 'jupyter', 'tensorflow', 'pytorch']
        }
        # Flatten the skills dictionary for skill extraction
        self.all_skills = [skill for category in self.skills_dict.values() for skill in category]
        
    def preprocess_text(self, text):
        """Clean and preprocess text data."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)
    
    def extract_skills(self, text):
        """Extract skills from text based on predefined skill dictionary."""
        text = text.lower()
        found_skills = []
        
        # Check for each skill in the text
        for skill in self.all_skills:
            # Use word boundary to avoid partial matches
            if re.search(r'\b' + re.escape(skill) + r'\b', text):
                found_skills.append(skill)
                
        return found_skills
    
    def categorize_skills(self, skills_list):
        """Categorize skills into different domains."""
        categorized = {}
        for category, skills in self.skills_dict.items():
            categorized[category] = [skill for skill in skills_list if skill in skills]
        return categorized
    
    def calculate_skill_match_score(self, resume_skills, job_skills):
        """Calculate match score based on skills overlap."""
        if not job_skills:
            return 0.0
        
        matches = set(resume_skills).intersection(set(job_skills))
        return len(matches) / len(job_skills)
    
    def fit_transform(self, resumes_df, job_description):
        """Process resumes and job description and calculate match scores."""
        # Preprocess job description
        processed_job = self.preprocess_text(job_description)
        job_skills = self.extract_skills(job_description)
        
        # Add processed text to DataFrame
        resumes_df['processed_text'] = resumes_df['resume_text'].apply(self.preprocess_text)
        
        # Extract skills from resumes
        resumes_df['skills'] = resumes_df['resume_text'].apply(self.extract_skills)
        resumes_df['skill_match_score'] = resumes_df['skills'].apply(
            lambda x: self.calculate_skill_match_score(x, job_skills)
        )
        
        # Categorize skills
        resumes_df['categorized_skills'] = resumes_df['skills'].apply(self.categorize_skills)
        
        # Combine all documents for TF-IDF
        documents = resumes_df['processed_text'].tolist()
        documents.append(processed_job)
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Calculate cosine similarity
        similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
        resumes_df['tfidf_similarity'] = similarity_scores
        
        # Calculate combined score (weighted average of TF-IDF and skill-based scoring)
        resumes_df['combined_score'] = (0.6 * resumes_df['tfidf_similarity'] + 
                                        0.4 * resumes_df['skill_match_score'])
        
        resumes_df['rank'] = resumes_df['combined_score'].rank(ascending=False)
        
        return resumes_df, job_skills
    
    def get_top_matches(self, df, threshold=0.2):
        """Get resumes that match above the threshold."""
        return df[df['combined_score'] >= threshold].sort_values('combined_score', ascending=False)
    
    def visualize_results(self, df, job_skills):
        """Create visualizations for the matching results."""
        # Set up the plotting area
        plt.figure(figsize=(15, 10))
        
        # 1. Bar chart of combined scores
        plt.subplot(2, 2, 1)
        sorted_df = df.sort_values('combined_score', ascending=False)
        sns.barplot(x='resume_id', y='combined_score', data=sorted_df)
        plt.title('Combined Match Scores by Resume')
        plt.xlabel('Resume ID')
        plt.ylabel('Match Score')
        
        # 2. Comparative bar chart: TF-IDF vs Skill scores
        plt.subplot(2, 2, 2)
        chart_data = pd.melt(sorted_df, 
                             id_vars=['resume_id'],
                             value_vars=['tfidf_similarity', 'skill_match_score'],
                             var_name='Score Type', 
                             value_name='Score')
        sns.barplot(x='resume_id', y='Score', hue='Score Type', data=chart_data)
        plt.title('Comparison: TF-IDF vs Skill Match Scores')
        plt.xlabel('Resume ID')
        plt.ylabel('Score')
        
        # 3. Heatmap of skill categories by resume
        plt.subplot(2, 2, 3)
        # Create a matrix of skill categories
        category_data = []
        for _, row in df.iterrows():
            row_data = {}
            for category, skills in row['categorized_skills'].items():
                row_data[category] = len(skills)
            category_data.append(row_data)
            
        category_df = pd.DataFrame(category_data, index=df['resume_id'])
        sns.heatmap(category_df, annot=True, cmap='YlGnBu', fmt='d')
        plt.title('Skill Categories by Resume')
        
        # 4. Word cloud of job skills (using a simple bar chart instead)
        plt.subplot(2, 2, 4)
        job_skill_freq = Counter(job_skills)
        skill_df = pd.DataFrame.from_dict(job_skill_freq, orient='index').reset_index()
        skill_df.columns = ['Skill', 'Count']
        skill_df = skill_df.sort_values('Count', ascending=False).head(10)
        sns.barplot(x='Count', y='Skill', data=skill_df)
        plt.title('Top Skills Required in Job Description')
        
        plt.tight_layout()
        plt.savefig('resume_matching_analysis.png')
        plt.show()
        
        return 'resume_matching_analysis.png'
    
    def generate_report(self, df, job_skills, threshold=0.2):
        """Generate a detailed report of the matching results."""
        matches = self.get_top_matches(df, threshold)
        
        print("\n===== RESUME MATCHING REPORT =====")
        print(f"Job requires {len(job_skills)} skills: {', '.join(job_skills)}")
        print(f"\nFound {len(matches)} matching resumes above threshold {threshold}:")
        
        for _, match in matches.iterrows():
            print(f"\nResume ID: {match['resume_id']}")
            print(f"Combined Match Score: {match['combined_score']:.2f}")
            print(f"TF-IDF Similarity: {match['tfidf_similarity']:.2f}")
            print(f"Skill Match Score: {match['skill_match_score']:.2f}")
            print(f"Ranking: #{int(match['rank'])}")
            
            # Print skills by category
            print("Skills by category:")
            for category, skills in match['categorized_skills'].items():
                if skills:
                    print(f"  - {category.replace('_', ' ').title()}: {', '.join(skills)}")
            
            # Calculate missing skills
            missing_skills = set(job_skills) - set(match['skills'])
            if missing_skills:
                print(f"Missing skills: {', '.join(missing_skills)}")
            else:
                print("All required skills are present!")
                
        return matches


# Sample resumes and job description data
data = {
    'resume_id': [1, 2, 3, 4, 5],
    'resume_text': [
        "Experienced data scientist with 5 years of skills in Python, machine learning, and data analysis. Proficient in SQL, TensorFlow, and neural networks. Worked on multiple NLP projects and predictive modeling.",
        "Software developer with expertise in Java, cloud computing, and project management. Knowledge of AWS, Docker, and Git. Bachelor's degree in Computer Science.",
        "Data analyst with proficiency in SQL, Python, and data visualization. Experience with Tableau and Power BI. Strong analytical skills and statistical knowledge.",
        "Full-stack developer with 4 years of experience. Skilled in JavaScript, React, Node.js, and MongoDB. Familiar with AWS and CI/CD pipelines using Jenkins.",
        "Machine learning engineer with expertise in deep learning models. Proficient in Python, PyTorch, and TensorFlow. Experience in computer vision and NLP projects."
    ]
}

job_description = """
Looking for a Senior Data Scientist skilled in Python, machine learning, SQL, and data analysis.
The ideal candidate will have experience with deep learning frameworks like TensorFlow or PyTorch,
natural language processing, and building predictive models. Knowledge of data visualization 
tools and cloud platforms (AWS/Azure) is a plus. Must be proficient in database technologies
and have a strong statistical background.
"""

# Initialize the system
matcher = ResumeMatchingSystem()

# Convert to DataFrame
df = pd.DataFrame(data)
print("Resumes loaded:", len(df))

# Process resumes and calculate matches
processed_df, job_skills = matcher.fit_transform(df, job_description)

# Generate detailed report
matcher.generate_report(processed_df, job_skills, threshold=0.3)

# Visualize the results
matcher.visualize_results(processed_df, job_skills)

# Display final DataFrame with all scores
print("\nFinal scoring details:")
print(processed_df[['resume_id', 'tfidf_similarity', 'skill_match_score', 'combined_score', 'rank']])