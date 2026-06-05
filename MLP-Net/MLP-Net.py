class MLPBlock(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.3):
        super(ResidualMLPBlock, self).__init__()

        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)

        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.activation(out)
        out = self.dropout(out)

        return out


class MLPNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.35):
        super(MLPNet, self).__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(28 * 28, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.block1 = ResidualMLPBlock(1024, 1024, dropout=dropout)
        self.block2 = ResidualMLPBlock(1024, 512, dropout=dropout)
        self.block3 = ResidualMLPBlock(512, 512, dropout=dropout)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.block4 = ResidualMLPBlock(256, 256, dropout=dropout)
        self.block5 = ResidualMLPBlock(256, 128, dropout=dropout)

        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        self.fc4 = nn.Linear(128, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
    
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward_features(self, x):
    
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.block4(x)
        x = self.block5(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc4(x)
        return x
