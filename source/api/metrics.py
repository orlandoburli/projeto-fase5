from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter('predict_requests_total', 'Total prediction requests')
REQUEST_LATENCY = Histogram('predict_request_duration_seconds', 'Latency of prediction requests')
MODEL_ACCURACY = Gauge('model_accuracy_score', 'Accuracy of trained model')
MODEL_AUC = Gauge('model_auc_score', 'ROC AUC score of trained model')
