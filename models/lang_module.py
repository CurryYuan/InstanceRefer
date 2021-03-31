from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.basic_blocks import *

class LangModule(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_text_classes = args.num_classes
        self.use_lang_classifier = args.use_lang_cls
        self.use_bidir = args.use_bidir

        self.word_projection = nn.Sequential(nn.Linear(args.embedding_size, args.word_output_dim),
                                             nn.ReLU(),
                                             nn.Dropout(args.word_dropout),
                                             nn.Linear(args.word_output_dim, args.word_output_dim),
                                             nn.ReLU())

        self.gru = nn.GRU(
            input_size=args.word_output_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            batch_first=True,
            bidirectional=self.use_bidir,
        )

        hidden_size = args.hidden_size if not self.use_bidir else args.hidden_size * 2
        self.fc_attribute = nn.Linear(hidden_size, 1)
        self.fc_cls = nn.Linear(hidden_size, 1)
        self.fc_relation = nn.Linear(hidden_size, 1)
        self.fc_scene = nn.Linear(hidden_size, 1)

        # language classifier
        if args.use_lang_cls:
            self.lang_cls = nn.Sequential(
                nn.Linear(args.word_output_dim, args.num_classes),
            )

    def rnn_encoding(self, embed, length, data_dict):
        embed = self.word_projection(embed)
        feats = pack_padded_sequence(embed, length, batch_first=True, enforce_sorted=False)
        feats, hidden = self.gru(feats)

        # Reshape *final* output to (batch_size, n_word, hidden_size)
        feats, _ = pad_packed_sequence(feats, batch_first=True)
        data_dict['lang_feat'] = feats

        mask = length_to_mask(length, max_len=feats.shape[1])

        atten_a = self.fc_relation(feats).squeeze(2)
        atten_a = torch.softmax(atten_a, dim=1)  # (B, N)
        atten_a = atten_a * mask
        atten_a = atten_a / atten_a.sum(1, keepdim=True)
        embed_a = torch.bmm(atten_a.unsqueeze(1), embed[:, :atten_a.shape[1]]).squeeze(1)

        atten_cls = self.fc_cls(feats).squeeze(2)
        atten_cls = torch.softmax(atten_cls, dim=1)  # (B, N)
        atten_cls = atten_cls * mask
        atten_cls = atten_cls / atten_cls.sum(1, keepdim=True)
        embed_cls = torch.bmm(atten_cls.unsqueeze(1), embed[:, :atten_cls.shape[1]]).squeeze(1)

        atten_rel = self.fc_relation(feats).squeeze(2)
        atten_rel = torch.softmax(atten_rel, dim=1)  # (B, N)
        atten_rel = atten_rel * mask
        atten_rel = atten_rel / atten_rel.sum(1, keepdim=True)
        embed_rel = torch.bmm(atten_rel.unsqueeze(1), embed[:, :atten_rel.shape[1]]).squeeze(1)

        atten_scene = self.fc_scene(feats).squeeze(2)
        atten_scene = torch.softmax(atten_scene, dim=1)  # (B, N)
        atten_scene = atten_scene * mask
        atten_scene = atten_scene / atten_scene.sum(1, keepdim=True)
        embed_scene = torch.bmm(atten_scene.unsqueeze(1), embed[:, :atten_scene.shape[1]]).squeeze(1)

        data_dict['atten_attr'] = atten_a
        data_dict['atten_rel'] = atten_rel
        data_dict['atten_scene'] = atten_scene
        data_dict['lang_cls_feats'] = embed_cls
        data_dict['lang_attr_feats'] = embed_a
        data_dict['lang_rel_feats'] = embed_rel
        data_dict['lang_scene_feats'] = embed_scene

        return data_dict

    def forward(self, data_dict):
        """
        encode the input descriptions
        """

        feats = data_dict["lang_feat"]  # (B, N, C)
        length = data_dict["lang_len"]
        data_dict = self.rnn_encoding(feats, length, data_dict)  # (B, C)

        # classify
        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(data_dict["lang_cls_feats"])

        return data_dict

    def get_pharse(self, feat, tags, index):
        pharse = []
        temp = []
        state = False
        for i, tag in enumerate(tags):
            if tag == index:
                temp.append(feat[:, i])
                state = True
            elif state:
                state = False
                temp = torch.cat(temp, dim=0).sum(0, keepdim=True)
                pharse.append(temp)
                temp = []
        if len(pharse) > 0:
            pharse = torch.cat(pharse, dim=0)  # (n_pharse, dim)
        return pharse