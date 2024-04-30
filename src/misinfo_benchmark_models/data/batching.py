import typing
from operator import itemgetter

import torch


def collator(features: typing.List[typing.Dict]) -> typing.Dict[str, typing.Any]:
    batch = dict()

    batch["attention_mask"] = torch.stack(
        list(map(lambda x: itemgetter("attention_mask")(x), features)), axis=0
    )

    max_len = torch.max(torch.sum(batch["attention_mask"], axis=1))

    batch["input_ids"] = torch.stack(
        list(map(lambda x: itemgetter("input_ids")(x), features)), axis=0
    )

    batch["input_ids"] = batch["input_ids"][:max_len]
    batch["attention_mask"] = batch["attention_mask"][:max_len]

    #print(batch["input_ids"].shape, batch["attention_mask"].shape)

    batch["labels"] = torch.tensor(list(map(itemgetter("label"), features)))

    return batch

def static_collator(features: typing.List[typing.Dict]) -> typing.Dict[str, typing.Any]:
    batch = dict()

    batch["input_ids"] = torch.stack(
        list(map(lambda x: itemgetter("input_ids")(x), features)), axis=0
    )

    batch["attention_mask"] = torch.stack(
        list(map(lambda x: itemgetter("attention_mask")(x), features)), axis=0
    )

    batch["labels"] = torch.tensor(list(map(itemgetter("label"), features)))

    return batch