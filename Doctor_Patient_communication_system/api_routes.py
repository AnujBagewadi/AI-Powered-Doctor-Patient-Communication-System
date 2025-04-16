# api_routes.py
from flask import Blueprint, jsonify, request
from flask_login import current_user, login_required
from models import User, MedicalRecord, Discussion, Comment
from app import db
from datetime import datetime

api = Blueprint('api', __name__)

@api.route('/patients', methods=['GET'])
@login_required
def get_patients():
    if not current_user.is_doctor:
        return jsonify({'error': 'Unauthorized access'}), 403
    
    patients = User.query.filter_by(is_doctor=False).all()
    
    patients_data = []
    for patient in patients:
        last_record = MedicalRecord.query.filter_by(user_id=patient.id).order_by(MedicalRecord.record_date.desc()).first()
        
        patients_data.append({
            'id': patient.id,
            'username': patient.username,
            'email': patient.email,
            'last_visit': last_record.record_date.isoformat() if last_record else None,
            'record_count': len(patient.medical_records)
        })
    
    return jsonify({'patients': patients_data})

@api.route('/recent_records', methods=['GET'])
@login_required
def get_recent_records():
    if not current_user.is_doctor:
        return jsonify({'error': 'Unauthorized access'}), 403
    
    records = db.session.query(
        MedicalRecord, User.username.label('patient_name')
    ).join(
        User, MedicalRecord.user_id == User.id
    ).order_by(
        MedicalRecord.record_date.desc()
    ).limit(5).all()
    
    records_data = []
    for record, patient_name in records:
        records_data.append({
            'id': record.id,
            'patient_id': record.user_id,
            'patient_name': patient_name,
            'record_date': record.record_date.isoformat(),
            'diagnosis': record.diagnosis
        })
    
    return jsonify({'records': records_data})

@api.route('/recent_discussions', methods=['GET'])
@login_required
def get_recent_discussions():
    discussions = db.session.query(
        Discussion, 
        User.username.label('author_name'),
        db.func.count(Comment.id).label('comment_count')
    ).join(
        User, Discussion.user_id == User.id
    ).outerjoin(
        Comment, Discussion.id == Comment.discussion_id
    ).group_by(
        Discussion.id
    ).order_by(
        Discussion.date_posted.desc()
    ).limit(5).all()
    
    discussions_data = []
    for discussion, author_name, comment_count in discussions:
        discussions_data.append({
            'id': discussion.id,
            'title': discussion.title,
            'content': discussion.content,
            'date_posted': discussion.date_posted.isoformat(),
            'author_name': author_name,
            'comment_count': comment_count
        })
    
    return jsonify({'discussions': discussions_data})

@api.route('/medical_record/<int:record_id>', methods=['GET'])
@login_required
def get_medical_record(record_id):
    record = MedicalRecord.query.get_or_404(record_id)
    
    # Check if user has access to this record
    if not current_user.is_doctor and record.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized access'}), 403
    
    patient = User.query.get(record.user_id)
    
    record_data = {
        'id': record.id,
        'patient_id': record.user_id,
        'patient_name': patient.username,
        'record_date': record.record_date.isoformat(),
        'diagnosis': record.diagnosis,
        'prescription': record.prescription,
        'notes': record.notes,
        'summary': record.summary
    }
    
    return jsonify({'record': record_data})

@api.route('/patient/<int:user_id>/records', methods=['GET'])
@login_required
def get_patient_records(user_id):
    # Check if user has access to this patient's records
    if not current_user.is_doctor and user_id != current_user.id:
        return jsonify({'error': 'Unauthorized access'}), 403
    
    records = MedicalRecord.query.filter_by(user_id=user_id).order_by(MedicalRecord.record_date.desc()).all()
    
    records_data = []
    for record in records:
        records_data.append({
            'id': record.id,
            'record_date': record.record_date.isoformat(),
            'diagnosis': record.diagnosis,
            'summary': record.summary
        })
    
    return jsonify({'records': records_data})