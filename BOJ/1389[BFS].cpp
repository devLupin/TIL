#include <iostream>
#include <cstring>
#include <vector>
#include <queue>
using namespace std;

bool edge[101][101];
bool visited[101];

int n, m;

typedef struct {
	int node, w;
}pos;

int bfs(int src, int dest)
{
	memset(visited, false, sizeof(visited));

	queue<pos> q;
	q.push({ src,0 });
	visited[src] = true;

	while (!q.empty()) {
		pos cur = q.front();
		q.pop();

		if (cur.node == dest)
			return cur.w;

		for (int dir = 1; dir <= n; dir++) {
			if (!visited[dir] && edge[cur.node][dir]) {
				q.push({ dir, cur.w + 1 });
				visited[dir] = true;
			}
		}
	}
	return -1;
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	memset(edge, false, sizeof(edge));

	cin >> n >> m;
	for (int i = 0; i < m; i++) {
		int s, d; cin >> s >> d;
		edge[s][d] = edge[d][s] = true;
	}

	vector<int> v;

	for (int i = 1; i <= n; i++) {
		int temp = 0;
		for (int j = 1; j <= n; j++) {
			if (i != j)
				temp += bfs(i, j);
		}
		v.push_back(temp);
	}

	int node = 0, min = v[0];
	for (int i = 0; i < v.size(); i++)
		if (min > v[i]) {
			node = i;
			min = v[i];
		}

	cout << node + 1;
}