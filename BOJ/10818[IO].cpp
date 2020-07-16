#include<iostream>
#include<string>
#include<vector>
#include<algorithm>
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int n; cin >> n;
	vector<int> v;
	for (int i = 0; i < n; i++) {
		int temp; cin >> temp;
		v.push_back(temp);
	}
	sort(v.begin(), v.end());
	cout << v[0] << ' ' << v[v.size() - 1];
}