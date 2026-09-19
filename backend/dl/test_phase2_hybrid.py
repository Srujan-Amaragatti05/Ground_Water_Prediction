import os
import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
from phase2_hybrid import build_hybrid_model, get_hybrid_data

class TestPhase2Hybrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv_path = 'mock_dataset_hybrid.csv'
        data = {
            'STATE_UT': ['S1']*12 + ['S2']*7,
            'DISTRICT': ['D1']*12 + ['D2']*7,
            'BLOCK': ['B1']*12 + ['B2']*7,
            'VILLAGE': ['V1']*12 + ['V2']*7,
            'LATITUDE': [10.0]*12 + [20.0]*7,
            'LONGITUDE': [10.0]*12 + [20.0]*7,
            'Date': [
                '2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01',
                '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01', '2024-01-01',
                '2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01', '2024-01-01'
            ],
            'WL(mbgl)': np.random.rand(19)*10,
            'Temperature': np.random.rand(19)*30 + 10,
            'Humidity': np.random.rand(19)*50 + 30,
            'Rainfall': np.random.rand(19)*100,
            'LAT_API': [10.0]*19,
            'LON_API': [10.0]*19
        }
        pd.DataFrame(data).to_csv(cls.csv_path, index=False)
        cls.data, cls.scalers, cls.annual_df = get_hybrid_data(cls.csv_path)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.csv_path):
            os.remove(cls.csv_path)

    def test_preprocessing_and_shapes(self):
        self.assertEqual(len(self.data['X_train'].shape), 3)
        self.assertEqual(self.data['X_train'].shape[1], 5)
        self.assertEqual(self.data['X_train'].shape[2], 4)
        
        self.assertEqual(len(self.data['S_train'].shape), 2)
        self.assertEqual(self.data['S_train'].shape[1], 2)
        
        self.assertEqual(len(self.data['y_train'].shape), 2)
        self.assertEqual(self.data['y_train'].shape[1], 1)
        
    def test_train_val_test_separation(self):
        test_targets = self.data['test_target_years']
        for ty in test_targets:
            self.assertIn(ty, [2022, 2023])

    def test_hybrid_model_build_and_shapes(self):
        model = build_hybrid_model((5, 4), (2,))
        self.assertTrue(model.count_params() > 0)

    def test_predictions_no_nan(self):
        model = build_hybrid_model((5, 4), (2,))
        if len(self.data['X_val']) > 0:
            preds = model.predict([self.data['X_val'], self.data['S_val']])
            self.assertFalse(np.isnan(preds).any())
            self.assertFalse(np.isinf(preds).any())
        
    def test_model_save_reload(self):
        model = build_hybrid_model((5, 4), (2,))
        save_path = 'temp_test_hybrid_model.keras'
        model.save(save_path)
        loaded_model = tf.keras.models.load_model(save_path)
        if len(self.data['X_val']) > 0:
            preds_original = model.predict([self.data['X_val'], self.data['S_val']])
            preds_loaded = loaded_model.predict([self.data['X_val'], self.data['S_val']])
            np.testing.assert_allclose(preds_original, preds_loaded, rtol=1e-5)
        if os.path.exists(save_path):
            os.remove(save_path)

if __name__ == '__main__':
    unittest.main()
