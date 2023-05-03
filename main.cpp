#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <stack>

#define INF 999999
#define NIL (-1)

using namespace std;

/** Intending to implement the next algorithms:
 *  BFS, DFS, DIJKSTRA, BELLMAN_FORD, MOORE, KOSARAJU
 */

/**
 * @struct Edge: source_Node/Vertex, dest_Node/Vertex, weight
 */
struct Edge {
    int src, dest, weight;
};

class Graph {
private:
    /**
     * @param V: number of Nodes/Vertices
     * @param E: number of Edges
     */
    int V, E;
    /**
     * @param lstAdj: adjacency list for our Graph
     */
    vector<vector<pair<int, int>>> lstAdj;
    /**
     * @param lstAdjComplementary: complementary adjacency list for our Graph
     */
    vector<vector<pair<int, int>>> lstAdjComplementary;
    /**
     * @param matAdj: adjacency matrix for our Graph
     */
    int **matAdj;
    /**
     * @param visited: vector to retain the visited nodes/vertices and their corresponding related component
     */
    vector<int> visited;
    /**
     * @param edges: vector of edges
     */
    vector<Edge> edges;
    /**
     * @param nodeStack: stack for determining the related components in the KOSARAJU ALGORITHM
     */
    stack<int> nodeStack;
public:
    explicit Graph(const string &inputFile);

    ~Graph();

    void DFS(int vertex, int componentID, vector<vector<pair<int, int>>> lstAdjDFS);

    void BFS(int source);

    vector<int> DIJKSTRA(int source, vector<vector<pair<int, int>>> lstAdjD, int VD);

    vector<int> BELLMAN_FORD(int source, const vector<Edge>& edgesB, int VB);

    bool JOHNSON();

    void MOORE(int source, int destination);

    void KOSARAJU();
};

Graph::Graph(const string &inputFile) {
    // initializing the input file
    ifstream input(inputFile);

    // initializing the number of vertices and edges
    int nVertices, nEdges;
    input >> nVertices >> nEdges;
    this->V = nVertices;
    this->E = nEdges;

    // initializing the adjacency list
    this->lstAdj.resize(this->V);

    // initializing the complementary adjacency list
    this->lstAdjComplementary.resize(this->V);

    // initializing the adjacency matrix
    this->matAdj = new int *[this->V];
    for (int i = 0; i < this->V; i++) {
        this->matAdj[i] = new int[this->V];
        for (int j = 0; j < this->V; j++)
            this->matAdj[i][j] = 0;
    }

    // initializing the edges vector
    this->edges.resize(this->E);

    // populating the list, the matrix and the edges
    int src, dest, weight;
    for (int i = 0; i < this->E; i++) {
        input >> src >> dest >> weight;
        this->lstAdj[src].emplace_back(dest, weight);
        this->lstAdjComplementary[dest].emplace_back(src, weight);
        this->matAdj[src][dest] = 1;
        this->edges.emplace_back(src, dest, weight);
    }

    // closing the file
    input.close();
}

Graph::~Graph() {
    for (int i = 0; i < this->V; i++)
        delete[] this->matAdj[i];
    delete[] this->matAdj;
}

/*
 * DFS ALGORITHM
 * Builds a forest of nodes/vertices based on their related component
 * Complexity: O(V + E)
 *
 * - V: because the algorithm explores each node once
 * - E: because the algorithm explores each edge once based on the node's adjacency list
 */
void Graph::DFS(int vertex, int componentID, vector<vector<pair<int, int>>> lstAdjDFS) {
    // updating the visited for vertex with it's respective componentID
    this->visited[vertex] = componentID;
    // iterating through vertex's adjacency list
    for (const auto &neighbor: lstAdjDFS[vertex]) {
        // keeping the node from the neighbor
        int node = neighbor.first;
        // verifying if this node is visited or not
        if (this->visited[node] == 0)
            // then we are DFS'ing this node too
            DFS(node, componentID, lstAdjDFS);
    }
    this->nodeStack.push(vertex);
}

/*
 * BFS ALGORITHM
 * Builds a tree based on the source node/vertex
 * Complexity: O(V + E)
 *
 * - V: because the algorithm explores each node once
 * - E: because the algorithm explores each edge once based on the node's adjacency list
 */

void Graph::BFS(int source) {
    // initializing the visited vector with the value 0 -> no node was visited
    this->visited = vector<int>(V, 0);
    // initializing a vector of distances for printing the distances to every node
    vector<int> distances(V, INF);
    // initializing a queue for exploring each node that we are discovering
    queue<int> nodeQueue;
    // updating the visited and distance field for 'source' node -> here we start
    this->visited[source] = 1;
    distances[source] = 0;
    // pushing the source node into the queue to start processing the graph
    nodeQueue.push(source);

    while (!nodeQueue.empty()) {
        // extracting the current node from the queue
        int current = nodeQueue.front();
        nodeQueue.pop();
        // searching for neighbors in the adjacency list of the current node
        for (const auto &neighbor: this->lstAdj[current]) {
            // deciding if the node was already visited or not
            if (this->visited[neighbor.first]) {
                // if not, we are updating it's visited value, dist and pushing it into the queue
                this->visited[neighbor.first] = 1;
                distances[neighbor.first] = distances[current] + 1;
                nodeQueue.push(neighbor.first);
            }
        }
    }

    // printing the results
    for (auto i = 0; i < distances.size(); i++) {
        if (distances[i] == INF)
            cout << i << ": INF ";
        else
            cout << i << ": " << distances[i] << " ";
    }
    cout << endl;
}

/*
 * DIJKSTRA's ALGORITHM
 * Finds the shortest path from a source node to all the other nodes,
 * in a graph where there are no negative weights.
 * It implements a Greedy approach, processing the nodes based on their distances, increasing order.
 *
 * Complexity: O(V^2)
 * - V^2: because, if every node is adjancent with all the other nodes,
 *        the algorithm will check if it can relax the nodes V-1 times,
 *        in a loop executing V times
 */
vector<int> Graph::DIJKSTRA(int source, vector<vector<pair<int, int>>> lstAdjD, int VD) {
    // initializing the distance list
    vector<int> distances(VD, INF);
    // the source node will have its distance equal to 0 -> here we start
    distances[source] = 0;

    // initializing a priority queue which will contain
    // the nodes/vertices in increasing order based on the distance to them
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;

    // pushing the source node/vertex into the pq, with it's corresponding distance
    pq.emplace(source, distances[source]);

    // processing the graph until the priority queue has any nodes/vertices left
    while (!pq.empty()) {
        // extracting the most important node/vertex based on the lowest distance
        auto current = pq.top();
        pq.pop();

        // searching for its neighbors in its adjacency list
        for (const auto &neighbor: lstAdjD[current.first])
            // verifying if the minimum distance can be improved
            if (distances[neighbor.first] > distances[current.first] + neighbor.second) {
                // if so, we update the distance and then adding the neighbor node into the priority queue
                distances[neighbor.first] = distances[current.first] + neighbor.second;
                pq.emplace(neighbor.first, distances[neighbor.first]);
            }
    }
    // returning the solved distance vector
    return distances;
}

/*
 * BELLMAN_FORD ALGORITHM
 * Finds the shortest path from a source node to all the other nodes,
 * even if the weights are negative.
 *
 * If there are negative cycles in a graph:
 *         v.dist > u.dist + w(u, v) ; doesn't matter how many times we are processing the (u, v) edge
 *
 * The longest path from a node to another node may not exceed V - 1 nodes,
 * so the algorithm is processing the graph V - 1 times
 *
 * Complexity: O(V * E)
 * - V: because we are processing the graph V - 1 times
 * - E: because we are searching in the adjacency list of every node => total number of edges
 */
vector<int> Graph::BELLMAN_FORD(int source, const vector<Edge>& edgesB, int VB) {
    // initializing a distance vector
    vector<int> distances(VB, INF);
    // the distance to source will be 0 -> here we start
    distances[source] = 0;

    // here starts the processing of the graph
    for (auto i = 0; i < VB; i++)
        // processing each edge
        for (const auto &edge: edgesB) {
            int src = edge.src;
            int dest = edge.dest;
            int weight = edge.weight;

            // determining if the current edge may be relaxed
            if (distances[src] != INF && distances[dest] > distances[src] + weight) {
                // if so, we relax it
                distances[dest] = distances[src] + weight;
            }
        }

    // determining if the graph contains negative cycles
    // if so, we cannot determine a minimum distance path between the source node and the rest of nodes
    for (const auto &edge: edgesB) {
        int src = edge.src;
        int dest = edge.dest;
        int weight = edge.weight;

        // determining if we can relax the edge even more
        if (distances[src] != INF && distances[dest] > distances[src] + weight) {
            // if so, the graph contains negative cycles -> there doesn't exist a solution
            return {};
        }
    }
    // else there exists a solution
    return distances;
}

/*
 * JOHNSON'S ALGORITHM
 *
 * Finds the shortest path from every node to every other node in the graph,
 * even if the weights are negative.
 *
 * It uses both the BELLMAN_FORD and DIJKSTRA's algorithms.
 *
 * It uses a new added node which connects to every node in the graph,
 * with the new edges added having a weight of 0.
 *
 * It applies the BELLMAN_FORD algorithm on this new node added.
 *
 * It determines a positive weight based on the distances retrieved from the
 * BELLMAN_FORD algorithm, so that DIJKSTRA's algorithm could work.
 *
 *           w'(u, v) = w(u, v) + newDist[u] - newDist[v]
 *
 *                ---  The re-weight formula ---
 *
 * It then applies the DIJKSTRA's algorithm on every node,
 * determining the original weight after.
 *
 * Complexity: (V^2 * logV + V*E)
 * - V^2 * logV: from DIJKSTRA
 * - V*E : from BELLMAN_FORD
 */

bool Graph::JOHNSON() {
    // initializing a new_edges vector, to add all the new edges with the new source node added
    vector<Edge> new_edges = this->edges;

    // adding the new edges
    for (int i = 0; i < this->V; i++) {
        new_edges.emplace_back(this->V, i, 0);
    }

    // calculating the new potential distances using the BELLMAN_FORD algorithm
    vector<int> potentialDistances = this->BELLMAN_FORD(this->V, new_edges, this->V + 1);


    // checking if the BELLMAN_FORD algorithm returned a solution
    if (potentialDistances.empty()) {
        // if not, we exit
        cout << "No solution.\n";
        return false;
    }

    // re-weighting the edges
    for (int i = 0; i < this->V; i++)
        for (auto &neighbor: this->lstAdj[i]) {
            neighbor.second = neighbor.second + potentialDistances[i] - potentialDistances[neighbor.first];
            cout << i << " " << neighbor.first << " " << neighbor.second << endl;
        }

    // now we apply DIJKSTRA's algorithm on every node in the graph
    for (int i = 0; i < this->V; i++) {
        // determining the new distances with positive weights
        vector<int> newDist = this->DIJKSTRA(i, this->lstAdj, this->V);

        // printing the new distance
        for (int j = 0; j < V; j++) {
            if (newDist[j] >= INF)
                cout << "INF ";
            else {
                // recalculating the distance using the potential distances we calculated earlier
                newDist[j] = newDist[j] - potentialDistances[i] + potentialDistances[j];
                cout << newDist[j] << " ";
            }
        }
        cout << endl;
    }
    return true;
}

/*
 * MOORE's ALGORITHM
 *
 * Determines the shortest path from the source node to the destination node.
 *
 * It uses a BFS approach to determine the distances and parents of every node.
 *
 * Complexity: O(V*E) -> BFS's time complexity
 */
void Graph::MOORE(int source, int destination) {
    // initializing the distance and parent vector
    vector<int> distances(this->V, INF);
    vector<int> parents(this->V, NIL);

    // the source will have it's distance = 0 and parent = NIL -> here we start
    distances[source] = 0;
    parents[source] = NIL; // already done this, but for better visualisation

    // creating a queue for processing the nodes in the order of their discovery
    queue<int> nodeQueue;

    // adding the current node in the queue
    nodeQueue.push(source);

    // here we start processing the graph
    while(!nodeQueue.empty()) {
        // extracting the current node from the queue
        int current = nodeQueue.front();
        nodeQueue.pop();

        // we search for its neighbors in its adjacency list
        for(const auto& neighbor: this->lstAdj[current])
            // we check if it hasn't been visited yet
            if(distances[neighbor.first] == INF) {
                // if so, we update the distance, the parent and push the node into the queue
                distances[neighbor.first] = distances[current] + 1;
                parents[neighbor.first] = current;
                nodeQueue.push(neighbor.first);
            }
    }

    // now we check if the destination node has been visited or not
    if(distances[destination] == INF) {
        // if so, there doesn't exist any path between the two nodes
        cout << "No path between " << source << " and " << destination << endl;
        return;
    }

    // else, we start to construct the path
    stack<int> path;
    int current = destination;

    // we will be iterating through the parent's of the nodes till the reach the source node
    while(current != -1) {
        path.push(current);
        current = parents[current];
    }

    // now we print the path
    cout << "The shortest path between " << source << " and " << destination << " is: " << endl;
    while(!path.empty()) {
        int node = path.top();
        path.pop();

        cout << node << " ";
    }
    cout << endl;
}

/*
 * KOSARAJU's ALGORITHM
 *
 * Determines the total number of related components in a directed graph.
 *
 * It implements two DFS's algorithms -> one on the main graph, and one on it's complementary.
 *
 * First, it determines the related components on the main graph, putting all the visited nodes in a stack.
 * Then, it determines the number of related components using the complementary graph.
 *
 * Complexity:
 */

void Graph::KOSARAJU() {
    // initializing the visited vector
    this->visited = vector<int>(this->V, 0);

    // current componenentID
    int componentID = 0;

    // DFS'ing the main graph and determining the related components
    for(int i = 0; i < this->V; i++)
        // checking if the node is visited or not
        if(this->visited[i] == 0) {
            // if not, we update the componentID and DFS the node
            componentID++;
            DFS(i, componentID, this->lstAdj);
        }

    // now, having the nodes in the reverse order of discovery
    // we can discover the strongly connected components in our graph
    int number;
    number = 0;
    componentID = 0;

    // reinitializing for the second DFS
    this->visited = vector<int>(this->V, 0);

    // we process the stack now
    while(!this->nodeStack.empty()) {
        // extracting the current node from the stack
        int node = this->nodeStack.top();
        this->nodeStack.pop();

        // verifying if the node is visited
        if(visited[node] == 0) {
            // if not, we DFS it
            number++;
            componentID++;
            DFS(node, componentID, this->lstAdjComplementary);
        }
    }

    cout << "Number of strongly connected components is: " << number << endl;
}

int main() {
    string inputFile = "7-in.txt";
    Graph graph(inputFile);
    graph.KOSARAJU();
    graph.MOORE(0, 4);
    graph.JOHNSON();
    return 0;
}
