#include<iostream>
#include<string>
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int n, sum = 0; cin >> n;
	string str;
	cin >> str;
	for (auto it = str.begin(); it != str.end(); it++)
			sum += ((*it) - '0');
	cout << sum;
}