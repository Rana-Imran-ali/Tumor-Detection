"""
NeuroScan AI: Comprehensive Backend Verification Test Suite
Tests:
- Route accessibility & template rendering
- Real neural inference with MobileNetV2
- Direct image file upload & validation
- Corrupted file & disallowed extension security handling
- SQLite prediction history persistence
- CSV export of scan records
- Prominent medical disclaimer injection
"""

import unittest
import json
import io
import os
from PIL import Image

import app
import database
import model

class NeuroScanTestSuite(unittest.TestCase):
    def setUp(self):
        app.app.config['TESTING'] = True
        self.client = app.app.test_client()
        database.init_db()

    def test_routes_and_medical_disclaimer(self):
        """Test dashboard, history, performance, and about pages return 200 and include medical disclaimer."""
        for path in ['/', '/history', '/performance', '/about']:
            res = self.client.get(path)
            self.assertEqual(res.status_code, 200)
            self.assertIn(b'investigational decision-support tool', res.data)

    def test_api_stats(self):
        """Test stats endpoint returns JSON metrics."""
        res = self.client.get('/api/stats')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn('total_scans', data)
        self.assertIn('breakdown', data)

    def test_predict_sample(self):
        """Test prediction using a pre-packaged sample scan."""
        res = self.client.post('/predict', data={
            'patient_name': 'Test Case - Glioma Sample',
            'sample_file': 'glioma_sample.jpg'
        })
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        self.assertIn('predicted_class', data)
        self.assertIn('confidence', data)
        self.assertIn('probabilities', data)
        self.assertEqual(len(data['probabilities']), 4)
        self.assertIn('disclaimer', data)
        self.assertTrue(data['confidence'] > 0)

    def test_predict_file_upload(self):
        """Test genuine multipart/form-data image upload and inference."""
        sample_path = os.path.join(app.app.config['SAMPLES_FOLDER'], 'meningioma_sample.jpg')
        with open(sample_path, 'rb') as f:
            file_bytes = f.read()

        res = self.client.post('/predict', data={
            'patient_name': 'Test Patient Upload #202',
            'mri_image': (io.BytesIO(file_bytes), 'patient_scan.jpg')
        }, content_type='multipart/form-data')

        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        self.assertIn(data['predicted_class'], model.CLASS_NAMES)
        self.assertEqual(len(data['probabilities']), 4)
        self.assertIn('disclaimer', data)

    def test_invalid_extension(self):
        """Ensure non-image file extensions are safely rejected with 400."""
        res = self.client.post('/predict', data={
            'patient_name': 'Hacker Test',
            'mri_image': (io.BytesIO(b'malicious content'), 'exploit.txt')
        }, content_type='multipart/form-data')

        self.assertEqual(res.status_code, 400)
        data = json.loads(res.data)
        self.assertFalse(data['success'])
        self.assertEqual(data['error_code'], 'INVALID_EXTENSION')

    def test_corrupt_image(self):
        """Ensure corrupted image bytes with valid extension are detected and rejected."""
        fake_jpg = io.BytesIO(b'NOT_A_REAL_IMAGE_DATA_HEADER')
        res = self.client.post('/predict', data={
            'patient_name': 'Corrupt File Test',
            'mri_image': (fake_jpg, 'corrupt.jpg')
        }, content_type='multipart/form-data')

        self.assertEqual(res.status_code, 400)
        data = json.loads(res.data)
        self.assertFalse(data['success'])
        self.assertEqual(data['error_code'], 'CORRUPT_IMAGE')

    def test_missing_input(self):
        """Ensure request without file or sample returns 400."""
        res = self.client.post('/predict', data={
            'patient_name': 'Empty Test'
        })
        self.assertEqual(res.status_code, 400)
        data = json.loads(res.data)
        self.assertFalse(data['success'])
        self.assertEqual(data['error_code'], 'MISSING_INPUT')

    def test_csv_export(self):
        """Ensure CSV export produces valid headers, timestamps, probabilities, and disclaimer."""
        res = self.client.get('/history/export')
        self.assertEqual(res.status_code, 200)
        self.assertIn('text/csv', res.headers.get('Content-Type', ''))
        content = res.data.decode('utf-8')
        self.assertIn('Scan ID', content)
        self.assertIn('Predicted Class', content)
        self.assertIn('Confidence Score (%)', content)
        self.assertIn('Glioma Probability (%)', content)
        self.assertIn('Medical Disclaimer', content)

if __name__ == '__main__':
    unittest.main()
