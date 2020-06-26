#include <iostream>
#include <queue>
#include <cstring>
#include <vector>
using namespace std;

int map[51][51];

int n, m;
int r, c, d;

const int dy[] = { -1,0,1,0 };
const int dx[] = { 0,1,0,-1 };
const int back_dy[4] = { 1,0,-1,0 };
const int back_dx[4] = { 0,-1,0,1 };

int ans;

void dfs(int y, int x, int dir)
{
    if (map[y][x] == 1)
        return;
    if (map[y][x] == 0) {
        map[y][x] = 2;
        ans++;
    }

    for (int k = 0; k < 4; k++) {
        int next_dir = dir - 1 < 0 ? 3 : dir - 1;
        int ny = y + dy[next_dir];
        int nx = x + dx[next_dir];

        if (map[ny][nx] == 0) {
            dfs(ny, nx, next_dir);
            return;
        }
        else {
            dir = next_dir;
        }
    }

    int ny = y + back_dy[dir], nx = x + back_dx[dir];
    dfs(ny, nx, dir);
}

int main(void)
{
    ios::sync_with_stdio(0);
    cin.tie(0); cout.tie(0);

    cin >> n >> m >> r >> c >> d;
    for (int y = 0; y < n; y++)
        for (int x = 0; x < m; x++)
            cin >> map[y][x];

    dfs(r, c, d);
    cout << ans;
}