import traceback
import xgboost_model_kidney_cancer as xgb_model

try:
    xgb_model.main()
except Exception as e:
    print(f'Error occurred: {str(e)}')
    print('Traceback:')
    print(traceback.format_exc())