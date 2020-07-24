#include <iostream>
#include <queue>
using namespace std;

const int sz = 100001;
int n, k, map[sz] = { 0, };
bool visited[sz] = { false, };

typedef struct {
	int x, sec;
}pos;

int sec = 0, cnt = 0;
void bfs(int src, int dest)
{
	queue<pos> q;

	q.push({ src,0 });
	visited[src] = true;

	while (!q.empty()) {
		pos cur = q.front();
		q.pop();
		visited[cur.x] = true;	// 큐에서 pop할 때 방문표시가 핵심임.

		if (sec && sec == cur.sec && cur.x == dest)
			cnt++;
		if (!sec && cur.x == dest) {
			sec = cur.sec;
			cnt++;
		}

		if (cur.x - 1 >= 0 && !visited[cur.x - 1])
			q.push({ cur.x - 1, cur.sec + 1 });

		if (cur.x + 1 < sz && !visited[cur.x + 1])
			q.push({ cur.x + 1, cur.sec + 1 });

		if (cur.x * 2 < sz && !visited[cur.x * 2])
			q.push({ cur.x * 2,cur.sec + 1 });
	}
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	cin >> n >> k;
	bfs(n, k);
	cout << sec << '\n' << cnt << '\n';
}