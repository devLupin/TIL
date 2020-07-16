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
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < i; j++)
			cout << ' ';
		for (int j = 0; j < n*2 - (i*2) - 1; j++)
			cout << '*';
		cout << '\n';
	}
	for (int i = n-2; i >= 0; i--) {
		for (int j = 0; j < i; j++)
			cout << ' ';
		for (int j = 0; j < n * 2 - (i * 2) - 1; j++)
			cout << '*';
		cout << '\n';
	}
}