import torch
import torch.nn as nn

class FallDetectNet(nn.Module):
    """
    Bidirectional LSTM + Attention cho dữ liệu keypoints (SEQ_LEN, 99).
    Input: 33 landmarks × 3 (x, y, z) — không dùng visibility.
    """
    def __init__(self, input_size=99, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        # output của BiLSTM là hidden_size * 2
        self.attention  = nn.Linear(hidden_size * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, 99)
        lstm_out, _ = self.lstm(x)              # (batch, seq_len, 256)

        # Attention
        scores  = self.attention(lstm_out)      # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len, 1)
        context = (lstm_out * weights).sum(dim=1)  # (batch, 256)

        return self.classifier(context)         # (batch, 2)
