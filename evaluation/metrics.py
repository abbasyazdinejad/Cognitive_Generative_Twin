import torch
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
import torch.nn as nn


class FidelityMetrics:
    def __init__(self):
        pass
    
    def compute_dtw(self, real_sequences, generated_sequences):
        dtw_distances = []
        
        for i in range(len(real_sequences)):
            real_seq = real_sequences[i].reshape(-1, 1)
            gen_seq = generated_sequences[i].reshape(-1, 1)
            
            distance, _ = fastdtw(real_seq, gen_seq, dist=euclidean)
            dtw_distances.append(distance)
        
        return np.mean(dtw_distances), np.std(dtw_distances)
    
    def compute_mmd(self, real_data, generated_data, kernel='rbf', sigma=1.0):
        real_data = real_data.reshape(real_data.shape[0], -1)
        generated_data = generated_data.reshape(generated_data.shape[0], -1)
        
        def rbf_kernel(X, Y, sigma):
            X = torch.FloatTensor(X)
            Y = torch.FloatTensor(Y)
            
            XX = torch.sum(X * X, dim=1).unsqueeze(1)
            YY = torch.sum(Y * Y, dim=1).unsqueeze(0)
            XY = torch.mm(X, Y.t())
            
            distances = XX + YY - 2 * XY
            
            K = torch.exp(-distances / (2 * sigma ** 2))
            
            return K
        
        K_XX = rbf_kernel(real_data, real_data, sigma)
        K_YY = rbf_kernel(generated_data, generated_data, sigma)
        K_XY = rbf_kernel(real_data, generated_data, sigma)
        
        mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
        
        return mmd.item()
    
    def compute_discriminator_score(self, real_data, generated_data):
        class SimpleDiscriminator(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 1)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = torch.sigmoid(self.fc3(x))
                return x
        
        real_data = torch.FloatTensor(real_data.reshape(real_data.shape[0], -1))
        generated_data = torch.FloatTensor(generated_data.reshape(generated_data.shape[0], -1))
        
        input_dim = real_data.shape[1]
        discriminator = SimpleDiscriminator(input_dim)
        optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        real_labels = torch.ones(len(real_data), 1)
        fake_labels = torch.zeros(len(generated_data), 1)
        
        for epoch in range(50):
            optimizer.zero_grad()
            
            real_output = discriminator(real_data)
            fake_output = discriminator(generated_data)
            
            real_loss = criterion(real_output, real_labels)
            fake_loss = criterion(fake_output, fake_labels)
            
            loss = real_loss + fake_loss
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            real_pred = discriminator(real_data)
            fake_pred = discriminator(generated_data)
            
            real_accuracy = ((real_pred > 0.5).float() == real_labels).float().mean().item()
            fake_accuracy = ((fake_pred < 0.5).float() == (1 - fake_labels)).float().mean().item()
            
            overall_accuracy = (real_accuracy + fake_accuracy) / 2
        
        return overall_accuracy


class RobustnessMetrics:
    def __init__(self):
        pass
    
    def add_gaussian_noise(self, data, noise_level, fraction_features):
        noisy_data = data.copy()
        
        num_features = data.shape[-1]
        num_features_to_corrupt = int(num_features * fraction_features)
        
        for i in range(len(noisy_data)):
            features_to_corrupt = np.random.choice(num_features, num_features_to_corrupt, replace=False)
            
            for feature_idx in features_to_corrupt:
                noise = np.random.normal(0, noise_level * np.std(data[:, :, feature_idx]))
                noisy_data[i, :, feature_idx] += noise
        
        return noisy_data
    
    def dropout_channels(self, data, dropout_rate):
        dropped_data = data.copy()
        
        num_channels = data.shape[-1]
        num_channels_to_drop = int(num_channels * dropout_rate)
        
        channels_to_drop = np.random.choice(num_channels, num_channels_to_drop, replace=False)
        
        dropped_data[:, :, channels_to_drop] = 0
        
        return dropped_data
    
    def compute_relative_accuracy(self, clean_accuracy, noisy_accuracy):
        return noisy_accuracy / clean_accuracy if clean_accuracy > 0 else 0


class OperationalMetrics:
    def __init__(self):
        pass
    
    def compute_ttfj(self, timestamps, correct_actions):
        ttfj_values = []
        
        for i in range(len(timestamps)):
            if correct_actions[i]:
                ttfj = timestamps[i]
                ttfj_values.append(ttfj)
        
        if len(ttfj_values) == 0:
            return float('inf'), 0
        
        return np.mean(ttfj_values), np.std(ttfj_values)
    
    def compute_error_rate(self, predictions, ground_truth):
        errors = np.sum(predictions != ground_truth)
        total = len(predictions)
        
        error_rate = errors / total if total > 0 else 0
        
        return error_rate * 100
    
    def compute_escalation_quality(self, predicted_escalations, expert_escalations):
        agreement = np.sum(predicted_escalations == expert_escalations)
        total = len(predicted_escalations)
        
        quality = agreement / total if total > 0 else 0
        
        return quality * 100


class InterpretabilityMetrics:
    def __init__(self):
        pass
    
    def evaluate_clarity(self, text, min_length=20, max_length=500):
        if len(text) < min_length:
            return 1
        elif len(text) > max_length:
            return 3
        
        sentences = text.split('.')
        if len(sentences) < 2:
            return 2
        
        return 4
    
    def evaluate_relevance(self, text, keywords):
        relevance_score = 0
        
        for keyword in keywords:
            if keyword.lower() in text.lower():
                relevance_score += 1
        
        normalized_score = min(5, int((relevance_score / len(keywords)) * 5) + 1)
        
        return normalized_score
    
    def evaluate_usefulness(self, text, action_verbs):
        usefulness_score = 0
        
        for verb in action_verbs:
            if verb.lower() in text.lower():
                usefulness_score += 1
        
        normalized_score = min(5, int((usefulness_score / len(action_verbs)) * 5) + 1)
        
        return normalized_score
    
    def compute_cf_consistency(self, original_severity, new_severity, original_text, counterfactual_text):
        severity_change_mentioned = False
        
        severity_keywords = ['reduce', 'increase', 'lower', 'higher', 'decrease']
        for keyword in severity_keywords:
            if keyword in counterfactual_text.lower():
                severity_change_mentioned = True
                break
        
        if not severity_change_mentioned:
            return 0
        
        if original_severity > new_severity:
            expected_keywords = ['reduce', 'lower', 'decrease', 'less']
        else:
            expected_keywords = ['increase', 'higher', 'escalate', 'more']
        
        consistency = 0
        for keyword in expected_keywords:
            if keyword in counterfactual_text.lower():
                consistency = 1
                break
        
        return consistency


class EvaluationSuite:
    def __init__(self, config):
        self.config = config
        self.fidelity = FidelityMetrics()
        self.robustness = RobustnessMetrics()
        self.operational = OperationalMetrics()
        self.interpretability = InterpretabilityMetrics()
    
    def evaluate_fidelity(self, real_data, generated_data):
        dtw_mean, dtw_std = self.fidelity.compute_dtw(real_data, generated_data)
        mmd = self.fidelity.compute_mmd(real_data, generated_data)
        disc_acc = self.fidelity.compute_discriminator_score(real_data, generated_data)
        
        return {
            'dtw_mean': dtw_mean,
            'dtw_std': dtw_std,
            'mmd': mmd,
            'discriminator_accuracy': disc_acc
        }
    
    def evaluate_robustness(self, model, clean_data, noise_levels, dropout_rates):
        results = {
            'noise': {},
            'dropout': {}
        }
        
        for noise_level in noise_levels:
            for fraction in [0.05, 0.10, 0.15]:
                noisy_data = self.robustness.add_gaussian_noise(clean_data, noise_level, fraction)
                
                key = f'noise_{int(noise_level*100)}_{int(fraction*100)}'
                results['noise'][key] = noisy_data
        
        for dropout_rate in dropout_rates:
            dropped_data = self.robustness.dropout_channels(clean_data, dropout_rate)
            
            key = f'dropout_{int(dropout_rate*100)}'
            results['dropout'][key] = dropped_data
        
        return results
    
    def evaluate_operational(self, timestamps, actions, ground_truth, escalations, expert_escalations):
        ttfj_mean, ttfj_std = self.operational.compute_ttfj(timestamps, actions == ground_truth)
        error_rate = self.operational.compute_error_rate(actions, ground_truth)
        escalation_quality = self.operational.compute_escalation_quality(escalations, expert_escalations)
        
        return {
            'ttfj_mean': ttfj_mean,
            'ttfj_std': ttfj_std,
            'error_rate': error_rate,
            'escalation_quality': escalation_quality
        }
    
    def evaluate_interpretability(self, rationales, counterfactuals, keywords, action_verbs, severities):
        clarity_scores = []
        relevance_scores = []
        usefulness_scores = []
        cf_consistency_scores = []
        
        for rationale in rationales:
            clarity = self.interpretability.evaluate_clarity(rationale)
            relevance = self.interpretability.evaluate_relevance(rationale, keywords)
            usefulness = self.interpretability.evaluate_usefulness(rationale, action_verbs)
            
            clarity_scores.append(clarity)
            relevance_scores.append(relevance)
            usefulness_scores.append(usefulness)
        
        for i, cf in enumerate(counterfactuals):
            if i < len(severities):
                original_sev, new_sev = severities[i]
                consistency = self.interpretability.compute_cf_consistency(
                    original_sev, new_sev, rationales[i] if i < len(rationales) else '', cf
                )
                cf_consistency_scores.append(consistency)
        
        return {
            'clarity': (np.mean(clarity_scores), np.std(clarity_scores)),
            'relevance': (np.mean(relevance_scores), np.std(relevance_scores)),
            'usefulness': (np.mean(usefulness_scores), np.std(usefulness_scores)),
            'cf_consistency': np.mean(cf_consistency_scores) * 100 if len(cf_consistency_scores) > 0 else 0
        }