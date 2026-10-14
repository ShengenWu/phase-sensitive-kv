"""Curate the single-token animal / name pools for the phase evaluator.

Requirements (paper App. C.2): 64 animal keys + 64 human-name values, each exactly
one token under the Qwen3 tokenizer in the leading-space form used inside records
(" ant Alice\n"). Keys and values are disjoint token sets. Deterministic given the
hard-coded candidate lists; writes pools.json.
"""

from __future__ import annotations

import json
import os

from transformers import AutoTokenizer

DATA_ROOT = os.environ.get("RESCSA_DATA_DIR", "/root/rescsa/data")
TOK_DIR = f"{DATA_ROOT}/qwen3-base"
OUT = f"{DATA_ROOT}/pools.json"

ANIMALS = [
    "ant", "bee", "cat", "dog", "fox", "owl", "pig", "rat", "bat", "cow", "elk", "hen",
    "ram", "goat", "wolf", "bear", "bird", "bull", "deer", "fish", "frog", "goose",
    "hare", "lion", "mole", "seal", "crow", "duck", "hawk", "pony",
    "shark", "snake", "whale", "camel", "crab", "flea",
    "lamb", "moth", "sheep", "tiger", "turtle",
    "beetle", "horse", "koala", "leopard", "lizard", "llama",
    "monkey", "mouse", "parrot", "rabbit", "salmon", "spider", "squirrel",
    "chicken", "dolphin", "hamster", "lobster", "pigeon", "zebra", "falcon",
    "heron", "stork", "crane", "badger", "beaver", "bison", "buffalo", "cattle",
    "cheetah", "chinchilla", "cobra", "condor", "coyote", "cricket", "crocodile",
    "dachshund", "dingo", "eagle", "ferret", "flamingo", "gazelle", "gecko",
    "gerbil", "gibbon", "gnu", "grackle", "guanaco", "hedgehog",
    "ibis", "iguana", "jackal", "jellyfish", "kestrel", "ladybug", "lemur",
    "lynx", "mallard", "mammoth", "marmot", "mastiff", "meerkat",
    "mink", "muskrat", "newt", "nutria", "ocelot", "octopus",
    "opossum", "orangutan", "oryx", "osprey", "peacock", "pelican", "penguin",
    "piglet", "quail", "quokka", "reindeer", "rhino", "rooster",
    "sable", "salamander", "starling", "stoat", "tapir", "termite", "thrush",
    "toucan", "tuna", "unicorn", "vicuna", "vole", "vulture", "wombat",
    "woodpecker", "wren", "yak", "aardvark", "albatross", "alligator",
]

NAMES = [
    "Alice", "Alan", "Anna", "Adam", "Alex", "Amber", "Amy", "Arthur", "Barbara", "Ben",
    "Betty", "Bruce", "Carl", "Carol", "Cathy", "Charles", "Chris", "Clara", "Dale", "David",
    "Diana", "Diane", "Emma", "Eric", "Ethan", "Eugene", "Eva", "Frank", "Grace", "Helen",
    "Henry", "Ian", "Irene", "Jack", "Jane", "Jason", "Jerry", "Joan", "John", "Judith",
    "Julia", "Karen", "Karl", "Kevin", "Laura", "Linda", "Lisa", "Lucy", "Luke", "Mark",
    "Mary", "Molly", "Nancy", "Nina", "Olga", "Oscar", "Paul", "Peter", "Philip", "Rachel",
    "Ralph", "Roy", "Ryan", "Sam", "Sarah", "Scott", "Sean", "Susan", "Tina", "Tom",
    "Victor", "Walter", "Wendy", "Zoe", "Aaron", "Abel", "Agnes", "Aiden", "Albert", "Alvin",
    "Amanda", "Amos", "Angela", "Anita", "April", "Arnold", "Aston", "Austin", "Bella", "Bern",
    "Betsy", "Bill", "Billy", "Bobby", "Bonnie", "Boris", "Brian", "Brit", "Brook", "Caleb",
    "Calvin", "Cameron", "Casey", "Cecil", "Cedric", "Chad", "Chase", "Chester", "Chloe", "Cliff",
]


def main():
    tok = AutoTokenizer.from_pretrained(TOK_DIR)

    def single_token(word):
        ids = tok.encode(" " + word, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    animals, names = [], []
    for w in ANIMALS:
        tid = single_token(w)
        if tid is not None and w not in [a[0] for a in animals]:
            animals.append((w, tid))
        if len(animals) == 64:
            break
    for w in NAMES:
        if w.lower() in {a[0] for a in animals}:  # names must not collide with animal words
            continue
        tid = single_token(w)
        if tid is not None and tid not in {a[1] for a in animals} and w not in [n[0] for n in names]:
            names.append((w, tid))
        if len(names) == 64:
            break

    assert len(animals) == 64, f"only {len(animals)} single-token animals"
    assert len(names) == 64, f"only {len(names)} single-token names"
    animal_ids = {t for _, t in animals}
    name_ids = {t for _, t in names}
    assert not (animal_ids & name_ids), "token id collision between pools"

    with open(OUT, "w") as f:
        json.dump({"animals": animals, "names": names}, f, indent=1)
    print(f"pools saved: 64 animals, 64 names -> {OUT}")
    print("sample animals:", [w for w, _ in animals[:8]])
    print("sample names:  ", [w for w, _ in names[:8]])


if __name__ == "__main__":
    main()
