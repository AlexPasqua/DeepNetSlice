import os
import networkx as nx
from typing import Tuple, List


def _check_if_graphml(file: str):
    """ Checks if a file is a GraphML file (checking the extension)

    :param file: path to the file to be checked
    :raise ValueError: is case the file is not a GraphML file
    """
    if not file.endswith(".graphml"):
        raise ValueError("{} is not a GraphML file".format(file))


def _check_required_attributes(network: nx.Graph, required_node_attributes: Tuple[str, ...],
                               required_link_attributes: Tuple[str, ...], **admissible_values: tuple):
    """ Checks whether all the required attributes are present in the nodes and link of the network passed as argument

    :param network: network whose nodes and links have to be checked (if they contain all the required attributes)
    :param required_node_attributes: tuple containing all the required attributes for the network's nodes
    :param required_link_attributes: tuple containing all the required attributes for the network's links
    :param admissible_values: (optional) extra arguments where the name is an attribute name and the value is a tuple with the admissible values

    :raise AssertionError:
        - in case some nodes/links don't contain all the required parameters
        - in case some non admissible values are used for some arguments
    """
    # check graph
    if "E2ELatency" in network.graph.keys():
        assert network.graph['E2ELatency'] > 0

    # check nodes
    for node_id, node in network.nodes.items():
        # add an attribute to specify if a VNF has been placed onto the PSN (initialized as False)
        if "reqCPU" in node.keys():
            # 'reqCPU' is a mandatory argument for NSPR, so if it's present, the node doesn't belong to a PSN, but to a NSPR
            node['placed'] = -1
        else:
            # it means the node belongs to a PSN and not to a NSPR
            node['availCPU'] = node['CPUcap']
            node['availRAM'] = node['RAMcap']
        # check that all required attributes are present in the current node
        assert all(req_attrib in node.keys() for req_attrib in required_node_attributes)
        # check that - if the admissible values for a certain attribute are passed - the value of each attribute is admissible
        for attrib, value in node.items():
            assert value in admissible_values.get(attrib, (value,))
            if attrib in ("CPUcap", "RAMcap", "availCPU", "availRAM", "reqCPU", "reqRAM"):
                assert value >= 0

    # check edges
    for node_A, node_B in list(network.edges):
        cur_link_attribs = network.edges[node_A, node_B].keys()
        cur_link_values = network.edges[node_A, node_B].values()
        cur_link_attribs_values = zip(cur_link_attribs, cur_link_values)
        # check that all required attributes are present in the current link
        assert all(attrib in cur_link_attribs for attrib in required_link_attributes)
        # check that - if the admissible values for a certain attribute are passed - the value of each attribute is admissible
        for attrib, value in cur_link_attribs_values:
            assert value in admissible_values.get(attrib, (value,))
            if attrib in ("BWcap", "reqBW", "Latency", "reqLatency"):
                assert value >= 0
        # initialize resources availabilities if PSN
        if "BWcap" in cur_link_attribs:
            network.edges[node_A, node_B]['availBW'] = network.edges[node_A, node_B]['BWcap']


def read_psn(graphml_file: str) -> nx.Graph:
    """ Reads a GraphML file containing the definition of a PSN

    :param graphml_file: GraphML file containing the definition of the PSN
    :return: a networkx.Graph representing the PSN

    :raise ValueError: if "graphml_file" is not a GraphML file
    :raise AssertionError: if some required attributes of nodes and links are missing
    """
    _check_if_graphml(graphml_file)  # check if the file passed is a GraphML file

    # read the GraphML file and create a nx.Graph object
    psn = nx.read_graphml(path=graphml_file, node_type=int)

    # check that the attributes of the graph are correct
    _check_required_attributes(network=psn,
                               required_node_attributes=("NodeType", "CPUcap", "RAMcap"),
                               required_link_attributes=("BWcap",),
                               NodeType=("UAP", "router", "switch", "server"))
    return psn


def read_single_nspr(graphml_file: str) -> nx.Graph:
    """ Reads a single NSPR (network slice placement request)

    :param graphml_file: GraphML file with the definition of the NSPR
    :return: the NSPR as a networkx.Graph object

    :raise ValueError: if "graphml_file" is not a GraphML file
    :raise AssertionError: if some required attributes of nodes and links are missing
    """
    _check_if_graphml(graphml_file)  # check if the file passed is a GraphML file

    # read the GraphML file and create a nx.Graph object
    nspr = nx.read_graphml(path=graphml_file, node_type=int)

    # check that the attributes of the graph are correct
    _check_required_attributes(network=nspr,
                               required_node_attributes=("reqCPU", "reqRAM"),
                               required_link_attributes=("reqBW",))
    return nspr


def read_nsprs(nsprs_path: str) -> List[nx.Graph]:
    """ Reads all the NSPRs (network slice placement requests) in a directory

    :param nsprs_path: either path to the directory with the files defining a NSPR each or the path to a single NSPR
    :return: the list of the NSPRs present in the directory (nsprs_path)
    :raise ValueError: if nsprs_path is neither a directory nor a file
    """
    if os.path.isdir(nsprs_path):
        return [read_single_nspr(graphml_file) for graphml_file in os.listdir(nsprs_path)]
    elif os.path.isfile(nsprs_path):
        return [read_single_nspr(nsprs_path)]
    else:
        raise ValueError(f"{nsprs_path} is neither a directory nor a file")
