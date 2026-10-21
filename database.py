import sqlite3
import os
import json
from datetime import datetime

DATABASE_NAME = "neuroscan.db"

def get_db_connection():
    """Establish and return a connection to the SQLite database with dictionary cursor."""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the SQLite database and create the scans table if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            filename TEXT NOT NULL,
            image_path TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            probabilities TEXT,
            clinical_notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_scan(patient_name, filename, image_path, predicted_class, confidence, probabilities, clinical_notes=""):
    """
    Insert a new scan prediction record into the database.
    
    Args:
        patient_name (str): Identifier or name of patient
        filename (str): Uploaded original filename
        image_path (str): Relative path to stored image in static/uploads
        predicted_class (str): Class label with highest probability
        confidence (float): Confidence score (0.0 to 100.0)
        probabilities (dict): Dictionary mapping class names to percentage floats
        clinical_notes (str): Additional notes or summary
    
    Returns:
        int: ID of the newly inserted record
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    probabilities_json = json.dumps(probabilities)
    
    cursor.execute("""
        INSERT INTO scans (patient_name, filename, image_path, predicted_class, confidence, probabilities, clinical_notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (patient_name, filename, image_path, predicted_class, round(confidence, 2), probabilities_json, clinical_notes))
    
    scan_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return scan_id

def get_all_scans():
    """Retrieve all logged scans from newest to oldest."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM scans ORDER BY created_at DESC")
    rows = cursor.fetchall()
    
    # Convert rows to standard dicts and parse probabilities JSON
    scans = []
    for row in rows:
        scan_dict = dict(row)
        try:
            scan_dict['probabilities'] = json.loads(scan_dict['probabilities']) if scan_dict['probabilities'] else {}
        except Exception:
            scan_dict['probabilities'] = {}
        scans.append(scan_dict)
        
    conn.close()
    return scans

def get_scan_by_id(scan_id):
    """Retrieve a single scan record by primary key."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM scans WHERE id = ?", (scan_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        scan_dict = dict(row)
        try:
            scan_dict['probabilities'] = json.loads(scan_dict['probabilities']) if scan_dict['probabilities'] else {}
        except Exception:
            scan_dict['probabilities'] = {}
        return scan_dict
    return None

def delete_scan(scan_id):
    """Delete a scan record by ID and attempt to clean up image file if exists."""
    scan = get_scan_by_id(scan_id)
    if not scan:
        return False
        
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
    conn.commit()
    conn.close()
    
    # Try deleting the image file if it's in static/uploads
    try:
        if scan.get('image_path') and not scan['image_path'].startswith('static/samples/'):
            full_path = os.path.join(os.path.dirname(__file__), scan['image_path'].replace('/', os.sep))
            if os.path.exists(full_path):
                os.remove(full_path)
    except Exception as e:
        print(f"Warning: could not delete file {scan.get('image_path')}: {e}")
        
    return True

def get_statistics():
    """Calculate summary counts for dashboard metrics."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as total FROM scans")
    total = cursor.fetchone()['total']
    
    cursor.execute("SELECT predicted_class, COUNT(*) as count FROM scans GROUP BY predicted_class")
    breakdown = {row['predicted_class']: row['count'] for row in cursor.fetchall()}
    conn.close()
    
    return {
        "total_scans": total,
        "breakdown": breakdown
    }

def export_scans_to_csv():
    """
    Exports all scan history from SQLite database as an in-memory CSV string.
    Includes patient info, predicted classes, confidence, 4-class breakdown, and medical disclaimer.
    """
    import io
    import csv

    scans = get_all_scans()
    output = io.StringIO()
    writer = csv.writer(output)

    # Write CSV Header
    writer.writerow([
        "Scan ID",
        "Patient ID / Name",
        "Original Filename",
        "Image Path",
        "Predicted Class",
        "Confidence Score (%)",
        "Glioma Probability (%)",
        "Meningioma Probability (%)",
        "No Tumor Probability (%)",
        "Pituitary Probability (%)",
        "Recorded Timestamp",
        "Clinical Notes",
        "Medical Disclaimer"
    ])

    disclaimer_text = "NeuroScan AI is an educational/research decision-support tool, not a certified clinical diagnostic device."

    for s in scans:
        probs = s.get("probabilities", {})
        writer.writerow([
            s.get("id"),
            s.get("patient_name"),
            s.get("filename"),
            s.get("image_path"),
            s.get("predicted_class"),
            s.get("confidence"),
            probs.get("Glioma", 0.0),
            probs.get("Meningioma", 0.0),
            probs.get("No Tumor", 0.0),
            probs.get("Pituitary", 0.0),
            s.get("created_at"),
            s.get("clinical_notes", ""),
            disclaimer_text
        ])

    return output.getvalue()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
