from .chain import Chain
from .text import Text


def get_model_dict(thing):
    if isinstance(thing, Chain):
        if thing.compiled:
            raise ValueError("Not implemented for compiled markovify.Chain")
        return thing.model
    if isinstance(thing, Text):
        if thing.chain.compiled:
            raise ValueError("Not implemented for compiled markovify.Chain")
        return thing.chain.model
    if isinstance(thing, list):
        print("get_model_dict : ")
        print(thing)
        return dict(thing)
    if isinstance(thing, dict):
        return thing

    raise ValueError(
        "`models` should be instances of list, dict, markovify.Chain, or markovify.Text"
    )


def combine(models, weights=None):
    if weights is None:
        weights = [1 for _ in range(len(models))]

    if len(models) != len(weights):
        raise ValueError("`models` and `weights` lengths must be equal.")

    model_dicts = list(map(get_model_dict, models))
    state_sizes = [len(list(md.keys())[0]) for md in model_dicts]

    if len(set(state_sizes)) != 1:
        raise ValueError("All `models` must have the same state size.")

    if len(set(map(type, models))) != 1:
        raise ValueError("All `models` must be of the same type.")

    c = {}

    for m, w in zip(model_dicts, weights):
        for state, options in m.items():
            current = c.get(state, {})
            for subseq_k, subseq_v in options.items():
                subseq_prev = current.get(subseq_k, 0)
                current[subseq_k] = subseq_prev + (subseq_v * w)
            c[state] = current

    ret_inst = models[0]

    if isinstance(ret_inst, Chain):
        return Chain.from_json(c)
    if isinstance(ret_inst, Text):
        ret_inst.find_init_states_from_chain.cache_clear()
        if any(m.retain_original for m in models):
            combined_sentences = []
            for m in models:
                if m.retain_original:
                    combined_sentences += m.parsed_sentences
            return ret_inst.from_chain(c, parsed_sentences=combined_sentences)
        else:
            return ret_inst.from_chain(c)
    if isinstance(ret_inst, list):
        return list(c.items())
    if isinstance(ret_inst, dict):
        return c

def generate_sentence(model, min_chars=8, max_chars=16, tries=100):
    candidate = model.make_short_sentence(
        max_chars, tries=tries
    ).replace(" ", "")
    while len(candidate) < min_chars:
            candidate = generate_sentence(model)
    return candidate
