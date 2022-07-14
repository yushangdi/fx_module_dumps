import glob
import importlib
import torch.fx as fx
from os.path import exists, abspath
from torch.fx.experimental.proxy_tensor import make_fx
import torch
from functorch._src.compilers import get_inputs

model_folder = "huggingface_graphs"


def draw_graph(traced_graph, name):
    from functorch.compile import draw_graph
    draw_graph(traced_graph, f'torch_figures/{name}')

def try_cse(fx_module, name):
    from functorch._src.compile_utils import fx_graph_cse
    new_graph = fx_graph_cse(fx_module.graph)
    if len(new_graph.nodes) != len(fx_module.graph.nodes):
        print(name)
        print(f"Reduced from {len(fx_module.graph.nodes)} to {len(new_graph.nodes)}")
        print(set([node.name for node in fx_module.graph.nodes]) - set([node.name for node in new_graph.nodes]))

def try_cat_opt(fx_module, name):
    import torch.fx
    print(name)
    for node in fx_module.graph.nodes:
        if node.target == torch.ops.aten.cat.default:
            if len(node.users) == 1:
                print("node args: ", node, node.args)
                user = tuple(node.users)[0]
                print("node user", user, user.args)
                while (len(user.users) == 1):
                    user = tuple(user.users)[0]
                    print(user, user.args)
                print()

    print()


def main():
    global model_folder
    assert exists(model_folder), f"{model_folder} does not exist"
    paths = glob.glob(f"{model_folder}/*/*")
    for dir in paths:
        path = dir.split('/')
        model_name = path[-1]
        module_path = '.'.join(path)
        input_data_path = f'{dir}/{model_name}.input'

        module = importlib.import_module(module_path)
        mod = module.FxModule()
        m = lambda inputs: mod(*inputs)
        try:
            print("Generating inputs for", model_name)
            inputs = get_inputs(input_data_path)

            print("Generating inputs for", model_name)
            traced_graph = make_fx(m)(inputs)
            traced_graph.graph.set_codegen(torch.fx.graph.CodeGen())  # avoid recursive pytree

            print("Running for", model_name)
            traced_graph(inputs)

            print("Anlysis for", model_name)
            # draw_graph(traced_graph, model_name)
            # try_cse(traced_graph, model_name)
            try_cat_opt(traced_graph, model_name)

            print("============================")
            
        except Exception as e:
            print(path)
            print(e)
            continue


if __name__ == "__main__":
    main()