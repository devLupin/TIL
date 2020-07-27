#include <iostream>
#include <algorithm>
using namespace std;

const int INF = 0x3f3f3f;
int n, m, h, a, b, ret = INF, visited[34][34];

// 매우 아름다운 탐색법 check().
bool check() {
    for (int i = 1; i <= n; i++) {
        int start = i;
        for (int j = 1; j <= h; j++) {
            if (visited[j][start])
                start++;
            else if (visited[j][start - 1])
                start--;
        }
        if (start != i) return false;
    }
    return true;
}
void make_ladder(int here, int cnt) {
    if (cnt > 3) return;

    if (check()) {
        ret = min(ret, cnt); 
        return;
    }
    for (int i = here; i <= h; i++) {
        for (int j = 1; j <= n; j++) {
            if (visited[i][j] || visited[i][j - 1] || visited[i][j + 1]) 
                continue;
            
            visited[i][j] = 1;
            make_ladder(i, cnt + 1);
            visited[i][j] = 0;
        }
    }
}
int main() {
    ios_base::sync_with_stdio(0);
    cin.tie(0); cout.tie(0);

    cin >> n >> m >> h;
    for (int i = 0; i < m; i++) {
        cin >> a >> b; 
        visited[a][b] = 1;
    }

    make_ladder(1, 0);
    
    cout << ((ret == INF) ? -1 : ret);
    return 0;
}