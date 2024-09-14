# -*- coding:utf8 -*-

NAME_DICT = {
    "conv": lambda name, cin, cout, kernel, stride, padding, _: "__".join([
        name, str(cin), str(cout), str(kernel), str(stride), str(padding)
    ]),
    "final": lambda name, in_shape, out_shape, _: "__".join([
        name, str(in_shape), str(out_shape)
    ]),
    "max_pool": lambda name, kernel: "__".join([name, str(kernel)]),
    "avg_pool": lambda name, kernel: "__".join([name, str(kernel)]),
    "skip": lambda name, cin, cout: "__".join([name, str(cin), str(cout)]),

    # resnet
    "residual_layer": lambda name, cin, cout, blocks, stride, concat: "__".join([
        name, str(cin), str(cout), str(blocks), str(stride), str(concat)
    ]),
    "residual_fc": lambda name, in_shape, out_shape: "__".join([
        name, str(in_shape), str(out_shape)
    ]),
    "flexible_residual_layer": lambda name, cin, cout, blocks, kernel, stride, conv: "__".join([
        name, str(cin), str(cout), str(blocks), str(kernel), str(stride), conv
    ]),
    "dbr": lambda name, cin, cout, kernel, stride: "__".join([
        name, str(cin), str(cout), str(kernel), str(stride)
    ]),
}


def get_lut_name(name, item_type, condition_dict):
    """
    Estimator和Layer使用condition_dict的方式来生成key_name
    考虑到面向不同的数据集，还是需要将输入尺寸和Lut表项关联起来

    arch_config:input_size 决定生成的lut.yaml里的input_size
    supernet:input_size 决定搜索lut时使用的input_size
    :param name:
    :param item_type:
    :param condition_dict:
    :return:
    """
    assert isinstance(condition_dict, dict)
    assert name in LutItem.TYPE_MAP.keys()
    origin_size = condition_dict.get("input_size")

    if item_type == LutItem.CONV:
        input_channel = condition_dict.get("input_channel")
        output_channel = condition_dict.get("output_channel")
        kernel = condition_dict.get("kernel")
        stride = condition_dict.get("stride")
        padding = condition_dict.get("padding")
        key_name = "__".join(
            [name, str(input_channel), str(output_channel), str(kernel), str(stride), str(padding)]).lower()

    elif item_type == LutItem.FINAL:
        input_shape = condition_dict.get("input_shape")
        output_shape = condition_dict.get("output_shape")
        key_name = "__".join([name, str(input_shape), str(output_shape)]).lower()

    elif item_type in [LutItem.MAX_POOL, LutItem.AVG_POOL]:
        kernel = condition_dict.get("kernel")
        key_name = "__".join([name, str(kernel)]).lower()

    elif item_type == LutItem.SKIP:
        input_channel = condition_dict.get("input_channel")
        output_channel = condition_dict.get("output_channel")
        key_name = "__".join([name, str(input_channel), str(output_channel)]).lower()

    # resnet18
    elif item_type == LutItem.RESIDUAL_LAYER:
        input_channel = condition_dict.get("input_channel")
        output_channel = condition_dict.get("output_channel")
        blocks_num = condition_dict.get("blocks")
        stride = condition_dict.get("stride")
        concat = condition_dict.get("concat")
        key_name = "__".join([
            name, str(input_channel), str(output_channel), str(blocks_num), str(stride), str(concat)
        ]).lower()

    elif item_type == LutItem.RESIDUAL_FC:
        input_shape = condition_dict.get("input_shape")
        output_shape = condition_dict.get("output_shape")
        key_name = "__".join([name, str(input_shape), str(output_shape)]).lower()

    elif item_type == LutItem.FLEXIBLE_RESIDUAL_LAYER:
        input_channel = condition_dict.get("input_channel")
        output_channel = condition_dict.get("output_channel")
        blocks_num = condition_dict.get("blocks")
        stride = condition_dict.get("stride")
        kernel = condition_dict.get("kernel")
        conv = condition_dict.get("conv")
        key_name = "__".join([
            name, str(input_channel), str(output_channel), str(blocks_num), str(kernel), str(stride), conv
        ]).lower()

    elif item_type == LutItem.DBR:
        input_channel = condition_dict.get("input_channel")
        output_channel = condition_dict.get("output_channel")
        kernel = condition_dict.get("kernel")
        stride = condition_dict.get("stride")
        key_name = "__".join([name, str(input_channel), str(output_channel), str(kernel), str(stride)]).lower()

    else:
        raise Exception("Invalid item type")

    return key_name.lower() + f"__{origin_size}*{origin_size}"


def mk_lut_name(line, input_size):
    """
    面向builder
    :param line:
    :param input_size: size of input  data
    :return:
    """
    line_name = line[0]
    if line_name not in NAME_DICT.keys():
        raise Exception(f"Name <{line_name}> invalid!")
    if len(line) > NAME_DICT[line_name].__code__.co_argcount:
        line = line[: NAME_DICT[line_name].__code__.co_argcount]
    primitive = NAME_DICT[line_name](*line)
    name = f"{primitive}__{input_size}*{input_size}"
    return name.lower()


class LutItem(object):
    """
    Conv means : Conv2dBNReLU
    Skip: Identity

    BN, Pool should be included by a total block
    """
    CONV = 1
    FINAL = 2
    SKIP = 3
    AVG_POOL = 4
    MAX_POOL = 5

    RESIDUAL_LAYER = 91
    RESIDUAL_FC = 92
    FLEXIBLE_RESIDUAL_LAYER = 93
    DBR = 94

    TYPE_MAP = {
        "conv": CONV,
        "skip": SKIP,
        "avg_pool": AVG_POOL,
        "max_pool": MAX_POOL,
        "final": FINAL,

        # resnet18
        "residual_layer": RESIDUAL_LAYER,
        "residual_fc": RESIDUAL_FC,
        "flexible_residual_layer": FLEXIBLE_RESIDUAL_LAYER,
        "dbr": DBR,
    }

    REVERSE = {
        tp: name for name, tp in TYPE_MAP.items()
    }

    def __init__(self, name: str, content: dict):
        """
        time: ms
        mem: MB
        name:
        conv__cin__cout__kernel__stride__padding
        final__cin__cout
        skip
        avg_pool__kernel
        max_pool__kernel

        :param name:
        :param content:
        """
        self.name = name
        self.time = content["time"]
        self.memory = content["memory"]

    def get_time(self):
        return self.time

    def get_memory(self):
        return self.memory
