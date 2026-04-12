import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from sentinel.classifier import RadicalClassifier
from sentinel.constants import LABEL_MAP


class TestRadicalClassifierMocked:
    @pytest.fixture
    def mock_torch_device(self):
        return MagicMock()

    @pytest.fixture
    def classifier_with_mocks(self):
        with patch('sentinel.classifier.AutoModelForSequenceClassification') as mock_model, \
             patch('sentinel.classifier.AutoTokenizer') as mock_tokenizer, \
             patch('sentinel.classifier.torch') as mock_torch:
            
            type(mock_torch).cuda = MagicMock(return_value=False)
            type(mock_torch).no_grad = MagicMock(return_value=MagicMock())
            type(mock_torch).softmax = MagicMock()
            type(mock_torch).argmax = MagicMock()
            mock_device = MagicMock()
            mock_torch.device.return_value = mock_device
            
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            mock_model_instance = MagicMock()
            mock_model.from_pretrained.return_value = mock_model_instance
            
            logits = MagicMock()
            probabilities = MagicMock()
            probabilities.__getitem__ = MagicMock(return_value=MagicMock(item=MagicMock(return_value=0.85)))
            probabilities.cpu.return_value.numpy.return_value = [0.1, 0.2, 0.3, 0.4]
            logits.__getitem__ = MagicMock(return_value=MagicMock(
                argmax=MagicMock(return_value=MagicMock(item=MagicMock(return_value=1))),
                softmax=MagicMock(return_value=probabilities)
            ))
            
            mock_outputs = MagicMock()
            mock_outputs.logits = logits
            mock_model_instance.return_value = mock_outputs
            
            classifier = RadicalClassifier(lazy_load=True)
            classifier._is_loaded = True
            classifier.tokenizer = mock_tokenizer_instance
            classifier.model = mock_model_instance
            classifier.device = mock_device
            
            yield classifier, mock_model_instance, mock_tokenizer_instance

    def test_classifier_lazy_load_flag(self):
        classifier = RadicalClassifier(lazy_load=True)
        assert classifier._is_loaded is False
        assert classifier.is_loaded is False

    def test_classifier_eager_load_flag(self):
        with patch('sentinel.classifier.AutoModelForSequenceClassification') as mock_model, \
             patch('sentinel.classifier.AutoTokenizer') as mock_tokenizer, \
             patch('sentinel.classifier.torch') as mock_torch:
            
            type(mock_torch).cuda = MagicMock(return_value=False)
            mock_device = MagicMock()
            mock_torch.device.return_value = mock_device
            mock_tokenizer.from_pretrained.return_value = MagicMock()
            mock_model.from_pretrained.return_value = MagicMock()
            
            classifier = RadicalClassifier(lazy_load=False)
            assert classifier._is_loaded is True

    def test_ensure_model_loaded_loads_when_not_loaded(self):
        with patch('sentinel.classifier.AutoModelForSequenceClassification') as mock_model, \
             patch('sentinel.classifier.AutoTokenizer') as mock_tokenizer, \
             patch('sentinel.classifier.torch') as mock_torch:
            
            type(mock_torch).cuda = MagicMock(return_value=False)
            mock_device = MagicMock()
            mock_torch.device.return_value = mock_device
            mock_tokenizer.from_pretrained.return_value = MagicMock()
            mock_model.from_pretrained.return_value = MagicMock()
            
            classifier = RadicalClassifier(lazy_load=True)
            classifier._ensure_model_loaded()
            assert classifier._is_loaded is True

    def test_ensure_model_loaded_skips_when_already_loaded(self):
        classifier = RadicalClassifier(lazy_load=True)
        classifier._is_loaded = True
        
        with patch('sentinel.classifier.AutoModelForSequenceClassification') as mock_model:
            classifier._ensure_model_loaded()
            mock_model.from_pretrained.assert_not_called()

    def test_warmup_calls_dummy_inference(self, classifier_with_mocks):
        classifier, mock_model, mock_tokenizer = classifier_with_mocks
        
        logits = MagicMock()
        probabilities = MagicMock()
        probabilities.__getitem__ = MagicMock(return_value=MagicMock(item=MagicMock(return_value=0.85)))
        probabilities.cpu.return_value.numpy.return_value = [0.1, 0.2, 0.3, 0.4]
        
        mock_outputs = MagicMock()
        mock_outputs.logits = logits
        mock_model.return_value = mock_outputs
        
        with patch.object(classifier, '_dummy_inference') as mock_inference:
            classifier.warmup(num_inferences=1)
            assert mock_inference.call_count == 3

    def test_predict_returns_correct_structure(self, classifier_with_mocks):
        classifier, mock_model, mock_tokenizer = classifier_with_mocks
        
        logits = MagicMock()
        probabilities = MagicMock()
        probabilities.__getitem__ = MagicMock(return_value=MagicMock(item=MagicMock(return_value=0.85)))
        probs_array = [0.1, 0.2, 0.3, 0.4]
        probabilities.cpu.return_value.numpy.return_value = probs_array
        
        predicted_class = 1
        probabilities_list = [MagicMock(item=MagicMock(return_value=probs_array[predicted_class]))]
        probabilities.__getitem__ = MagicMock(side_effect=lambda i: probabilities_list[i] if i < len(probabilities_list) else MagicMock())
        
        logits.__getitem__ = MagicMock(return_value=MagicMock(
            argmax=MagicMock(return_value=MagicMock(item=MagicMock(return_value=predicted_class))),
        ))
        type(logits).softmax = MagicMock(return_value=probabilities)
        
        mock_outputs = MagicMock()
        mock_outputs.logits = logits
        mock_model.return_value = mock_outputs
        
        with patch('sentinel.classifier.torch.no_grad') as mock_no_grad:
            mock_no_grad.return_value.__enter__ = MagicMock()
            mock_no_grad.return_value.__exit__ = MagicMock()
            
            result = classifier.predict("Test text")
            
            assert 'label' in result
            assert 'label_id' in result
            assert 'confidence' in result
            assert 'probabilities' in result
            assert result['label'] == LABEL_MAP[predicted_class]
            assert result['label_id'] == predicted_class

    def test_predict_batch_returns_correct_structure(self, classifier_with_mocks):
        classifier, mock_model, mock_tokenizer = classifier_with_mocks
        
        texts = ["Test one", "Test two"]
        
        mock_tokenizer.return_value = {'input_ids': MagicMock()}
        mock_outputs = MagicMock()
        mock_outputs.logits = MagicMock()
        mock_model.return_value = mock_outputs
        
        result = classifier.predict_batch(texts)
        
        assert isinstance(result, list)
        assert len(result) == len(texts)
        for item in result:
            assert 'text' in item
            assert 'label' in item
            assert 'label_id' in item
            assert 'confidence' in item
            assert 'probabilities' in item

    def test_get_fine_grained_scores_returns_dict(self, classifier_with_mocks):
        classifier, mock_model, mock_tokenizer = classifier_with_mocks
        
        mock_outputs = MagicMock()
        mock_outputs.logits = MagicMock()
        mock_model.return_value = mock_outputs
        
        result = classifier.get_fine_grained_scores("Test text")
        
        assert isinstance(result, dict)
        assert len(result) == 4

    def test_is_loaded_property(self, classifier_with_mocks):
        classifier, _, _ = classifier_with_mocks
        
        classifier._is_loaded = False
        assert classifier.is_loaded is False
        
        classifier._is_loaded = True
        assert classifier.is_loaded is True


class TestRadicalClassifierIntegration:
    def test_classifier_initializes_with_default_params(self):
        with patch('sentinel.classifier.AutoModelForSequenceClassification') as mock_model, \
             patch('sentinel.classifier.AutoTokenizer') as mock_tokenizer, \
             patch('sentinel.classifier.torch') as mock_torch:
            
            type(mock_torch).cuda = MagicMock(return_value=False)
            mock_device = MagicMock()
            mock_torch.device.return_value = mock_device
            mock_tokenizer.from_pretrained.return_value = MagicMock()
            mock_model.from_pretrained.return_value = MagicMock()
            
            classifier = RadicalClassifier()
            
            assert classifier.model_name == "distilbert-base-uncased"
            assert classifier.num_labels == 4

    def test_classifier_accepts_custom_checkpoint_path(self):
        with patch('sentinel.classifier.AutoModelForSequenceClassification') as mock_model, \
             patch('sentinel.classifier.AutoTokenizer') as mock_tokenizer, \
             patch('sentinel.classifier.torch') as mock_torch, \
             patch('sentinel.classifier.Path') as mock_path:
            
            type(mock_torch).cuda = MagicMock(return_value=False)
            mock_device = MagicMock()
            mock_torch.device.return_value = mock_device
            mock_tokenizer.from_pretrained.return_value = MagicMock()
            mock_model_instance = MagicMock()
            mock_model.from_pretrained.return_value = mock_model_instance
            
            mock_path_instance = MagicMock()
            mock_path.return_value = mock_path_instance
            mock_path_instance.exists.return_value = True
            mock_path_instance.iterdir.return_value = [MagicMock()]
            
            classifier = RadicalClassifier(checkpoint_path="models/checkpoint")
            
            assert classifier.checkpoint_path == "models/checkpoint"
