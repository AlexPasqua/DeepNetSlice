import os
import random
from typing import Tuple, List, Dict

import networkx as nx


def check_if_graphml(file: str):
    """ Checks if a file is a GraphML file (checking the extension)

    :param file: path to the file to be checked
    :raise ValueError: is case the file is not a GraphML file
    """
    if not file.endswith(".graphml"):
        raise ValueError("{} is not a GraphML file".format(file))


def _check_graph(network: nx.Graph):
    """ Checks that the graph is correct

    :param network: network that needs to be checked

    :raise AssertionError: if some graph's attributes are not correct
    """
    if "E2ELatency" in network.graph.keys():
        assert network.graph['E2ELatency'] > 0
        # if E2ELatency is present, it means the network is a NSPR
        if "ArrivalTime" in network.graph.keys():
            assert network.graph['ArrivalTime'] >= 0
        else:
            network.graph['ArrivalTime'] = 0
        if "DepartureTime" in network.graph.keys():
            assert network.graph['DepartureTime'] >= \
                   network.graph['ArrivalTime'] + len(network.nodes.keys())


def _check_nodes(network: nx.Graph, required_node_attributes: Tuple[str, ...],
                 **admissible_values: tuple):
    """ Checks that the nodes of the network are correct

    :param network: network whose nodes have to be checked
    :param required_node_attributes: tuple with all required attributes for the nodes
    :param admissible_values: (optional) extra arguments where the name is an
        attribute name and the value is a tuple with the admissible values

    :raise AssertionError:
        - in case some nodes don't contain all the required parameters
        - in case some non-admissible values are used for some arguments
    """
    for node_id, node in network.nodes.items():
        # if the admissible values for a certain attribute are passed,
        # check that the value of each attribute is admissible
        for attrib, value in node.items():
            assert value in admissible_values.get(attrib, (value,))
            if attrib in ("CPUcap", "RAMcap", "availCPU", "availRAM", "reqCPU", "reqRAM"):
                assert value >= 0
        # the following checks are for servers or VNFs only, in case skip
        if node.get("NodeType", "server") != "server":
            # if node hasn't attrib "NodeType", it's a VNF, so don't skip iteration
            continue
        if "reqCPU" in node.keys():
            # 'reqCPU' is a mandatory argument for NSPR, so if it's present, the node is a VNF
            # add an attribute to specify if a VNF has been placed onto the PSN
            node['placed'] = -1
        else:
            # it means the node belongs to a PSN and not to a NSPR
            node['availCPU'] = node['CPUcap']
            node['availRAM'] = node['RAMcap']
        # check that all required attributes are present in the current node
        assert all(req_attrib in node.keys() for req_attrib in required_node_attributes)


def _check_edges(network: nx.Graph, required_link_attributes: Tuple[str, ...], **admissible_values: tuple):
    """ Checks that the edges of the network are correct

    :param network: network whose edges have to be checked
    :param required_link_attributes: tuple with all required attributes for the links
    :param admissible_values: (optional) extra arguments where the name is an
        attribute name and the value is a tuple with the admissible values

    :raise AssertionError:
        - in case some links don't contain all the required parameters
        - in case some non-admissible values are used for some arguments
    """
    for node_A, node_B in list(network.edges):
        cur_link_attribs = network.edges[node_A, node_B].keys()
        cur_link_values = network.edges[node_A, node_B].values()
        cur_link_attribs_values = zip(cur_link_attribs, cur_link_values)
        # check that all required attributes are present in the current link
        assert all(attrib in cur_link_attribs for attrib in required_link_attributes)
        # if the admissible values for a certain attribute are passed,
        # check that the value of each attribute is admissible
        for attrib, value in cur_link_attribs_values:
            assert value in admissible_values.get(attrib, (value,))
            if attrib in ("BWcap", "reqBW", "Latency", "reqLatency"):
                assert value >= 0
        # initialize resources availabilities if PSN
        if "reqBW" in cur_link_attribs:
            # 'reqBW' is a mandatory argument for NSPR, so if it's present, the link is a VL
            network.edges[node_A, node_B]['placed'] = []
        else:
            # it means the link is physical and belongs to a PSN (and not to a NSPR)
            network.edges[node_A, node_B]['availBW'] = network.edges[node_A, node_B]['BWcap']


def check_required_attributes(network: nx.Graph, required_node_attributes: Tuple[str, ...],
                              required_link_attributes: Tuple[str, ...], **admissible_values: tuple):
    """ Checks whether all the required attributes are present in the nodes and link of the network passed as argument

    :param network: network whose nodes and links have to be checked
    :param required_node_attributes: tuple with all required attributes for the nodes
    :param required_link_attributes: tuple with all required attributes for the links
    :param admissible_values: (optional) extra arguments where the name is an
        attribute name and the value is a tuple with the admissible values

    :raise AssertionError:
        - in case some nodes/links don't contain all the required parameters
        - in case some non-admissible values are used for some arguments
    """
    _check_graph(network)
    _check_nodes(network, required_node_attributes, **admissible_values)
    _check_edges(network, required_link_attributes, **admissible_values)


def read_psn(graphml_file: str) -> nx.Graph:
    """ Reads a GraphML file containing the definition of a PSN

    :param graphml_file: GraphML file containing the definition of the PSN
    :return: a networkx.Graph representing the PSN

    :raise ValueError: if "graphml_file" is not a GraphML file
    :raise AssertionError: if some required attributes of nodes and links are missing
    """
    check_if_graphml(graphml_file)  # check if the file passed is a GraphML file

    # read the GraphML file and create a nx.Graph object
    psn = nx.read_graphml(path=graphml_file, node_type=int)

    # check that the attributes of the graph are correct
    check_required_attributes(network=psn,
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
    check_if_graphml(graphml_file)  # check if the file passed is a GraphML file

    # read the GraphML file and create a nx.Graph object
    nspr = nx.read_graphml(path=graphml_file, node_type=int)

    # check that the attributes of the graph are correct
    check_required_attributes(network=nspr,
                              required_node_attributes=("reqCPU", "reqRAM"),
                              required_link_attributes=("reqBW",))
    return nspr


def read_nsprs(nsprs_path: str) -> Dict[int, List[nx.Graph]]:
    """ Reads all the NSPRs (network slice placement requests) in a directory

    :param nsprs_path: either path to the directory with the files defining a
        NSPR each or the path to a single NSPR
    :return: a dict having as keys the arrival times of the NSPRs and as
        values the NSPRs themselves
    :raise ValueError: if nsprs_path is neither a directory nor a file
    """
    if not os.path.isdir(nsprs_path) and not os.path.isfile(nsprs_path):
        raise ValueError(f"{nsprs_path} is neither a directory nor a file")

    nspr_dict = {}  # save the NSPRs in a dict with the arrival times as keys
    if os.path.isfile(nsprs_path):
        nspr = read_single_nspr(nsprs_path)
        if nspr.graph['ArrivalTime'] not in nspr_dict.keys():
            nspr_dict[nspr.graph['ArrivalTime']] = [nspr]
        else:
            nspr_dict[nspr.graph['ArrivalTime']].append(nspr)
        return nspr_dict

    dir_path = nsprs_path
    for graphml_file in os.listdir(dir_path):
        nspr = read_single_nspr(os.path.join(dir_path, graphml_file))
        nspr_dict[nspr.graph['ArrivalTime']] = nspr_dict.get(nspr.graph['ArrivalTime'], []) + [nspr]
    return nspr_dict


def sample_nsprs(nsprs_path: str, n: int, min_arrival_time: int = 0,
                 max_duration: int = 100) -> Dict[int, List[nx.Graph]]:
    """ Samples a subset of NSPRs from a directory containing multiple NSPRs.
    It assigns random arrival and departure time to those NSPRs.

    :param nsprs_path: path to the directory containing the NSPRs
    :param n: number of NSPRs to sample
    :param min_arrival_time: minimum arrival time to assign to the sampled NSPRs
    :param max_duration: maximum duration (dep. time - arr. time) to assign to the sampled NSPRs
    :return: a dict having as keys the arrival times of the NSPRs and as
        values the NSPRs themselves
    :raise ValueError: if nsprs_path is not a directory
    """
    if not os.path.isdir(nsprs_path):
        raise ValueError(f"{nsprs_path} is not a directory")

    all_nsprs_files = os.listdir(nsprs_path)
    n = min(n, len(all_nsprs_files)) if n is not None else len(all_nsprs_files)
    sampled_nsprs_files = random.sample(all_nsprs_files, n)
    arrival_times = random.sample(range(min_arrival_time, min_arrival_time + max_duration), n)
    nspr_dict = {}
    for i, arr_time in enumerate(arrival_times):
        nspr = read_single_nspr(os.path.join(nsprs_path, sampled_nsprs_files[i]))
        nspr.graph['ArrivalTime'] = arr_time
        nspr.graph['duration'] = random.randint(len(nspr.nodes), max_duration)
        nspr_dict[arr_time] = nspr_dict.get(arr_time, []) + [nspr]
    return nspr_dict
