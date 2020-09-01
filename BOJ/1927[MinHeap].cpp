#include <iostream>
#include <queue>
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	priority_queue<int, vector<int>, greater<>> pq;
	int n; cin >> n;
	
	while (n--) {
		int x; cin >> x;
		
		if (x == 0) {
			if (pq.empty()) cout << 0 << '\n';
			else {
				cout << pq.top() << '\n';
				pq.pop();
			}
		}
		else pq.push(x);
	}
}
