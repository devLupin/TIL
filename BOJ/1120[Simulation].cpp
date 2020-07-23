#include <iostream>
#include <string>
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);
	
	string a, b;
	cin >> a >> b;

	int ans = 50, str_diff = b.size() - a.size();
	for (int i = 0; i <= str_diff; i++) {
		int comp = 0;
		for (int j = 0; j < a.size(); j++) {
			if (a[j] != b[i+j])
				comp++;
		}
		ans = (ans < comp) ? ans : comp;
	}

	cout << ans;
}