from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import nltk
import spacy
import re
from datetime import datetime
import json
from collections import defaultdict
import base64
import io
import os
from werkzeug.utils import secure_filename

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create upload folder if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Please install spacy model: python -m spacy download en_core_web_sm")
    # Create a dummy nlp object to prevent errors
    nlp = None

# In-memory storage (replace with MongoDB in production)
requirements_db = []
stakeholders_db = []
changes_log = []
brd_templates = []

class RequirementsExtractor:
    def __init__(self):
        self.requirement_patterns = [
            r"we need to\s+(.+?)(?:\.|$)",
            r"the system should\s+(.+?)(?:\.|$)",
            r"users? (?:should be able to|must|need to)\s+(.+?)(?:\.|$)",
            r"requirement is to\s+(.+?)(?:\.|$)",
            r"we want to\s+(.+?)(?:\.|$)",
            r"it should\s+(.+?)(?:\.|$)",
            r"must have\s+(.+?)(?:\.|$)",
            r"looking for\s+(.+?)(?:\.|$)"
        ]
        
        self.actor_keywords = ["user", "customer", "admin", "manager", "employee", "system", "client", "vendor"]
        self.action_verbs = ["create", "view", "update", "delete", "manage", "access", "generate", "submit", "approve", "track", "monitor", "search", "filter", "export", "import"]
    
    def extract_requirements(self, text):
        """Extract potential requirements from text"""
        text = text.lower()
        requirements = []
        
        # Pattern matching
        for pattern in self.requirement_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                req = match.group(1).strip()
                if len(req) > 10:  # Filter out very short matches
                    requirements.append(req)
        
        # NLP-based extraction if spaCy is available
        if nlp:
            try:
                doc = nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents]
                
                for sent in sentences:
                    sent_lower = sent.lower()
                    # Check if sentence contains action verbs and potential actors
                    if any(verb in sent_lower for verb in self.action_verbs):
                        if any(actor in sent_lower for actor in self.actor_keywords):
                            requirements.append(sent)
            except Exception as e:
                print(f"NLP extraction error: {e}")
        
        # Remove duplicates
        requirements = list(set(requirements))
        return requirements
    
    def generate_user_story(self, requirement):
        """Convert requirement to user story format"""
        # Default values
        actor = "user"
        action = requirement
        benefit = "improve the process"
        
        if nlp:
            try:
                doc = nlp(requirement)
                
                # Identify actor
                for token in doc:
                    if token.text.lower() in self.actor_keywords:
                        actor = token.text.lower()
                        break
                
                # Identify action
                for token in doc:
                    if token.pos_ == "VERB":
                        # Get the verb and its dependent words
                        verb_phrase = " ".join([t.text for t in token.subtree])
                        if len(verb_phrase) > 5:
                            action = verb_phrase
                            break
            except Exception as e:
                print(f"User story generation error: {e}")
        
        # Identify benefit (simplified - takes the part after "to" or "so that")
        benefit_patterns = [r"to\s+(.+)", r"so that\s+(.+)", r"in order to\s+(.+)"]
        for pattern in benefit_patterns:
            match = re.search(pattern, requirement, re.IGNORECASE)
            if match:
                benefit = match.group(1).strip()
                break
        
        # Format as user story
        user_story = f"As a {actor}, I want to {action} so that I can {benefit}"
        return user_story
    
    def generate_acceptance_criteria(self, user_story):
        """Generate acceptance criteria for a user story"""
        criteria = []
        
        # Extract the action from user story
        match = re.search(r"I want to (.+?) so that", user_story)
        if match:
            action = match.group(1)
            
            # Generate basic criteria
            criteria.append(f"GIVEN the {action.split()[0]} feature is available")
            criteria.append(f"WHEN the user attempts to {action}")
            criteria.append(f"THEN the system should successfully {action}")
            
            # Add validation criteria
            if any(word in action for word in ["create", "add", "submit"]):
                criteria.append("AND all required fields must be validated")
                criteria.append("AND success message should be displayed")
            
            if any(word in action for word in ["update", "edit", "modify"]):
                criteria.append("AND changes should be saved to the database")
                criteria.append("AND audit trail should be updated")
            
            if any(word in action for word in ["delete", "remove"]):
                criteria.append("AND confirmation dialog should be shown")
                criteria.append("AND related data should be handled appropriately")
        
        return criteria

class BRDGenerator:
    def __init__(self):
        self.template = """
# Business Requirements Document (BRD)

**Document Version:** 1.0  
**Date:** {date}  
**Author:** {author}  
**Project:** {project_name}

## Table of Contents
1. Executive Summary
2. Business Objectives
3. Scope
4. Functional Requirements
5. Non-Functional Requirements
6. User Stories
7. Process Flows
8. Data Requirements
9. Assumptions and Dependencies
10. Acceptance Criteria

---

## 1. Executive Summary
{executive_summary}

## 2. Business Objectives
{business_objectives}

## 3. Scope
### In Scope:
{in_scope}

### Out of Scope:
{out_scope}

## 4. Functional Requirements
{functional_requirements}

## 5. Non-Functional Requirements
### Performance Requirements:
- Response time should be less than 2 seconds
- System should support concurrent users
- 99.9% uptime requirement

### Security Requirements:
- All data must be encrypted
- Role-based access control
- Audit trail for all transactions

## 6. User Stories
{user_stories}

## 7. Process Flows
{process_flows}

## 8. Data Requirements
{data_requirements}

## 9. Assumptions and Dependencies
### Assumptions:
{assumptions}

### Dependencies:
{dependencies}

## 10. Acceptance Criteria
{acceptance_criteria}

---

## Appendix
### Diagrams and Mockups
{diagrams}

### Glossary
{glossary}
"""
    
    def generate_brd(self, requirements, project_info, diagrams=None):
        """Generate a complete BRD from requirements"""
        
        # Format requirements for BRD
        functional_reqs = []
        user_stories = []
        acceptance_criteria_all = []
        
        for i, req in enumerate(requirements, 1):
            functional_reqs.append(f"FR{i:03d}: {req['original_requirement']}")
            user_stories.append(f"**US{i:03d}:** {req['user_story']}")
            
            criteria_text = f"\n**For US{i:03d}:**\n"
            for criterion in req['acceptance_criteria']:
                criteria_text += f"- {criterion}\n"
            acceptance_criteria_all.append(criteria_text)
        
        # Generate BRD content
        brd_content = self.template.format(
            date=datetime.now().strftime("%Y-%m-%d"),
            author=project_info.get('author', 'Business Analyst'),
            project_name=project_info.get('project_name', 'Project Name'),
            executive_summary=project_info.get('executive_summary', 'This document outlines the business requirements for the project.'),
            business_objectives=project_info.get('business_objectives', '- Improve operational efficiency\n- Enhance user experience\n- Reduce manual processing time'),
            in_scope=project_info.get('in_scope', '- Core functionality implementation\n- User interface development\n- Integration with existing systems'),
            out_scope=project_info.get('out_scope', '- Third-party integrations\n- Mobile application\n- Advanced analytics'),
            functional_requirements='\n'.join(functional_reqs),
            user_stories='\n\n'.join(user_stories),
            process_flows=project_info.get('process_flows', '[Process flow diagrams to be added]'),
            data_requirements=project_info.get('data_requirements', '- User profile data\n- Transaction records\n- Audit logs'),
            assumptions=project_info.get('assumptions', '- Users have basic computer skills\n- Internet connectivity is available\n- Existing data can be migrated'),
            dependencies=project_info.get('dependencies', '- Database infrastructure\n- Authentication service\n- Email notification system'),
            acceptance_criteria='\n'.join(acceptance_criteria_all),
            diagrams='[Diagrams section - see attached images]' if diagrams else '[No diagrams provided]',
            glossary=project_info.get('glossary', 'BRD - Business Requirements Document\nUAT - User Acceptance Testing')
        )
        
        return brd_content

extractor = RequirementsExtractor()
brd_generator = BRDGenerator()

# Test route to verify server is working
@app.route('/test')
def test():
    return jsonify({'status': 'API is working', 'message': 'IRMS backend is running successfully!'})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_transcript', methods=['POST', 'OPTIONS'])
def analyze_transcript():
    if request.method == 'OPTIONS':
        # Handle preflight request
        return '', 200
    
    try:
        data = request.json
        transcript = data.get('transcript', '')
        
        if not transcript:
            return jsonify({'error': 'No transcript provided'}), 400
        
        # Extract requirements
        requirements = extractor.extract_requirements(transcript)
        
        if not requirements:
            return jsonify({
                'requirements': [],
                'stakeholders': [],
                'total_found': 0,
                'message': 'No requirements found. Try using phrases like "we need to", "the system should", "users must be able to"'
            })
        
        # Generate user stories and acceptance criteria
        results = []
        for i, req in enumerate(requirements[:5], 1):  # Limit to 5 for demo
            user_story = extractor.generate_user_story(req)
            acceptance_criteria = extractor.generate_acceptance_criteria(user_story)
            
            result = {
                'id': f'REQ-{len(requirements_db) + i:03d}',
                'original_requirement': req,
                'user_story': user_story,
                'acceptance_criteria': acceptance_criteria,
                'created_date': datetime.now().isoformat(),
                'status': 'Draft',
                'priority': 'Medium'
            }
            results.append(result)
        
        # Store in database
        requirements_db.extend(results)
        
        # Extract stakeholders
        stakeholders = extract_stakeholders(transcript)
        
        return jsonify({
            'requirements': results,
            'stakeholders': stakeholders,
            'total_found': len(requirements)
        })
    
    except Exception as e:
        print(f"Error in analyze_transcript: {str(e)}")
        return jsonify({'error': str(e)}), 500

def extract_stakeholders(text):
    """Extract potential stakeholders from text"""
    stakeholders = set()
    
    if nlp:
        try:
            doc = nlp(text)
            # Look for person names
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    stakeholders.add(ent.text)
        except Exception as e:
            print(f"Stakeholder extraction error: {e}")
    
    # Look for role mentions
    role_patterns = [
        r"(\w+\s+)?manager",
        r"(\w+\s+)?director",
        r"(\w+\s+)?lead",
        r"(\w+\s+)?team",
        r"product owner",
        r"business analyst",
        r"developer",
        r"tester"
    ]
    
    for pattern in role_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            stakeholders.add(match.group(0).strip().title())
    
    return list(stakeholders)

# New BRD-related routes
@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image uploads for brainstorming sessions"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'path': filepath
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_brd', methods=['POST'])
def generate_brd():
    """Generate a BRD document from requirements"""
    try:
        data = request.json
        requirement_ids = data.get('requirement_ids', [])
        project_info = data.get('project_info', {})
        include_diagrams = data.get('include_diagrams', False)
        
        # Get selected requirements
        selected_requirements = [req for req in requirements_db if req['id'] in requirement_ids]
        
        if not selected_requirements:
            return jsonify({'error': 'No requirements selected'}), 400
        
        # Generate BRD
        brd_content = brd_generator.generate_brd(selected_requirements, project_info, include_diagrams)
        
        # Save BRD
        brd_id = f"BRD-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        brd_filename = f"{brd_id}.md"
        brd_path = os.path.join('uploads', brd_filename)
        
        with open(brd_path, 'w') as f:
            f.write(brd_content)
        
        # Store BRD reference
        brd_record = {
            'id': brd_id,
            'filename': brd_filename,
            'created_date': datetime.now().isoformat(),
            'requirements': requirement_ids,
            'project_info': project_info
        }
        brd_templates.append(brd_record)
        
        return jsonify({
            'success': True,
            'brd_id': brd_id,
            'download_url': f'/download_brd/{brd_id}',
            'content_preview': brd_content[:500] + '...'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_brd/<brd_id>')
def download_brd(brd_id):
    """Download BRD as markdown file"""
    try:
        brd = next((b for b in brd_templates if b['id'] == brd_id), None)
        if not brd:
            return jsonify({'error': 'BRD not found'}), 404
        
        filepath = os.path.join('uploads', brd['filename'])
        return send_file(filepath, as_attachment=True, download_name=f"{brd_id}_BRD.md")
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_requirements', methods=['GET'])
def get_requirements():
    return jsonify(requirements_db)

@app.route('/update_requirement', methods=['POST'])
def update_requirement():
    try:
        data = request.json
        req_id = data.get('id')
        
        # Find requirement
        for req in requirements_db:
            if req['id'] == req_id:
                # Log change
                change = {
                    'requirement_id': req_id,
                    'timestamp': datetime.now().isoformat(),
                    'changes': []
                }
                
                # Update fields and track changes
                for field in ['user_story', 'status', 'priority']:
                    if field in data and data[field] != req[field]:
                        change['changes'].append({
                            'field': field,
                            'old_value': req[field],
                            'new_value': data[field]
                        })
                        req[field] = data[field]
                
                if change['changes']:
                    changes_log.append(change)
                
                return jsonify({'success': True, 'requirement': req})
        
        return jsonify({'error': 'Requirement not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_communication', methods=['POST'])
def generate_communication():
    try:
        data = request.json
        req_id = data.get('requirement_id')
        stakeholder_type = data.get('stakeholder_type', 'general')
        
        # Find requirement
        requirement = None
        for req in requirements_db:
            if req['id'] == req_id:
                requirement = req
                break
        
        if not requirement:
            return jsonify({'error': 'Requirement not found'}), 404
        
        # Generate communication template based on stakeholder type
        templates = {
            'executive': f"""
Subject: New Business Requirement - {requirement['id']}

Dear [Executive Name],

We have identified a new business requirement that aligns with our strategic objectives.

Summary: {requirement['user_story']}

Business Impact: This requirement will help improve operational efficiency and user satisfaction.

Priority: {requirement['priority']}
Target Timeline: [To be determined]

Please let me know if you need any additional information.

Best regards,
[Your Name]
""",
            'technical': f"""
Subject: Technical Requirement Specification - {requirement['id']}

Hi Team,

New requirement for implementation:

ID: {requirement['id']}
User Story: {requirement['user_story']}

Acceptance Criteria:
{chr(10).join(['- ' + criteria for criteria in requirement['acceptance_criteria']])}

Technical Considerations:
- API changes may be required
- Database schema updates needed
- Integration testing required

Please review and provide your estimates.

Thanks,
[Your Name]
""",
            'general': f"""
Subject: Requirement Update - {requirement['id']}

Hello,

We have documented a new requirement:

{requirement['user_story']}

Status: {requirement['status']}
Priority: {requirement['priority']}

Please feel free to reach out if you have any questions.

Best,
[Your Name]
"""
        }
        
        template = templates.get(stakeholder_type, templates['general'])
        
        return jsonify({'template': template})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_change_history', methods=['GET'])
def get_change_history():
    try:
        req_id = request.args.get('requirement_id')
        if req_id:
            history = [change for change in changes_log if change['requirement_id'] == req_id]
        else:
            history = changes_log
        
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting IRMS Flask server...")
    print("Make sure you have installed: pip install flask flask-cors nltk spacy")
    print("And downloaded: python -m spacy download en_core_web_sm")
    print("Server will be available at: http://localhost:5000")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
