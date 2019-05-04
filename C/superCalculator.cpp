#include <iostream>
#include <cstdlib>
#include <cmath>
#include<iomanip>
using namespace std;

class KajVec
{
public:

	KajVec() : size(2), elements(0)
	{
		arr = new int[size];
	}
	~KajVec()
	{
		delete[] arr;
	}

private:
	int *arr;
	int size;
	int elements;

public:
	int currentElements()const
	{
		return elements;
	}
	int& operator[](const int i)
	{
		return arr[i];
	}
	const int& operator[](const int i) const
	{
		return arr[i];
	}
	void pushback(const int& value)
	{
		resize();
		arr[elements] = value;
		++elements;
	}
	void pop()
	{
		if (elements > 0)
		{
			--elements;
		}
	}
	void insert(const int& value, const int index)
	{
		resize();
		if (elements > 0 && elements > index)
		{
			for (int j = elements; j > index; --j)
			{
				arr[j] = arr[j - 1];
			}
		}
		++elements;
		arr[index] = value;
	}

private:
	void resize()
	{
		if (elements + 1 <= size) return;
		size *= 2;
		int*arr2 = new int[size];
		for (int i = 0; i < elements; i++)
		{
			arr2[i] = arr[i];
		}
		delete[] arr;
		arr = arr2;
	}
};

void fill(const int n, KajVec& tab)
{
	int input;
	for (int i = 0; i < n; i++)
	{
		cin >> input;
		tab.pushback(input);
	}
}

int f0(const int n, const bool empty, KajVec& tab)
{
	if (empty)
	{
		fill(n, tab);
	}
	int min = tab[0];
	int pos = 0;
	for (int i = 0; i < tab.currentElements(); i++)
	{
		if (tab[i] < min)
		{
			min = tab[i];
			pos = i;
		}
	}
	if (empty) {
		KajVec	tab2;
		for (int i = 0; i < n; i++) {
			if (tab[i] == min) {
				tab2.pushback(i);

			}
		}
		for (int j = 0; j < tab2.currentElements(); j++) {
			cout << tab2[j] + 1 << " ";
		}
	}
	return pos;
}

void f1(const int n) {
	KajVec tab;
	fill(n, tab);
	KajVec tab2;

	for (int i = 0; i < n; i++) {
		int pos = f0(n, false, tab);
		tab2.insert(tab[pos], 0);
		tab[pos] = numeric_limits<int>::max();
	}

	for (int i = 0; i < n; i++) {
		cout << tab2[i] << " ";
	}
	cout << "\n";
}

double f2(const int n, KajVec& tab, const bool empty)
{
	if (empty) { fill(n, tab); }
	double S;
	double K;
	double G = 0;
	for (int i = 0; i < n; i++) {
		K = tab[i] * tab[i];
		S = G + K;
		G = S;
	}
	S = sqrt(S);
	return S;
}

double f3(const int n)
{
	KajVec tab;
	fill(n, tab);
	double y;
	double S = f2(n, tab, 0)*f2(n, tab, 0);
	double OS;
	double G = 0;
	double K;
	if (n == 1) {
		OS = 0;
	}
	else {
		for (int i = 0; i < n; i++) {
			K = tab[i];
			y = G + K;
			G = y;
		}
		y = y / n;
		OS = sqrt(S / n - (y*y));
	}
	cout << floor(OS) << endl;
	return OS;
}

int f4(const int n)
{
	
		KajVec tab;
		fill(n, tab);
	int temp;
	for (int i = 0; i < n / 2; ++i)
	{
		temp = tab[i];
		temp = tab[n - 1 - i];
		tab[n - 1 - i] = tab[i];
		tab[i] = temp;
	}
	for (int i = 0; i < n; ++i)
	{
		cout << tab[i] << " ";
	}

	return 0;

}

int f5(const int n)
{

	KajVec tab;
	fill(n, tab);
	int k;
	for (int i = 0; i < n; i++)
	{
		if (tab[i] > 5)
		{
			for (int j = 2; j < tab[i]; j++)
			{
				if (tab[i] % j == 0)
				{
					k = 1;
					break;
				}
				else { k = 0; }
			}
			if (k == 1)
			{
				cout << "0 ";
			}
			else if (k == 0)
			{
				cout << "1 ";
			}
		}

		else
		{
			if (tab[i] == 2 || tab[i] == 3 || tab[i] == 5)
			{
				cout << "1 ";
			}
			else
			{
				cout << "0 ";
			}
		}
	}
	std::cout << "\n";
	return 0;
}
struct points
{
	int x;
	int y;
};
int f6(const int n) {
	KajVec tab;
	fill(n, tab);

	return 0;
}

int f7(const int n) {
	double tab[4];
	double input;
	for (int i = 0; i < n; i++)
	{
		cin >> input;
		tab[i] = input;
	}
	double a = tab[0];
	double b = tab[1];
	double c = tab[2];
	double d = tab[3];
	double x1, x2, x3;
	double kkk = ((double)1.0) / (double(3.0));

	if (a == 0) {
		if (b == 0 && a == 0) {
			x1 = -d / c;
			cout << x1 << endl;
		}
		else {
			double delta;
			delta = (c*c) - (4 * b*d);

			if (delta > 0) {
				x1 = (-c - sqrt(delta)) / (2 * b);
				x2 = (-c + sqrt(delta)) / (2 * b);
				if (x2 > x1) {

					cout << floor(x1) << " " << floor(x2) << endl;
				}
				else { cout << floor(x2) << " " << floor(x1) << endl; }
			}
			else if (delta == 0) {
				x1 = -c / (2 * b);
				cout << floor(x1) << endl;

			}
			else {
				cout << "0" << endl;
			}
		}
	}
	else {
		double f, g, h, i, j, k, m, n, p;
		f = (c / a) - ((b*b) / (3 * a*a));
		g = ((2 * b*b*b) / (27 * a*a*a)) - (b*c / (3 * a*a)) + (d / a);
		h = (g*g / 4) + (f*f*f / 27);
		if (f == 0 && g == 0) {
			double kk = d / a;

			x1 = -pow(kk, kkk);
			x2 = -pow(kk, kkk);
			x3 = -pow(kk, kkk);
			cout << floor(x1) << " " << floor(x2) << " " << floor(x3) << endl;
		}
		else if (h > 0) {
			x1 = pow(((-g / 2) + sqrt(h)), kkk) - pow(((-g / 2) - sqrt(h)), kkk) - (b / (3 * a));
			cout << floor(x1) << endl;
		}
		else {
			i = sqrt(((g*g) / 4) - h);
			j = pow(i, kkk);
			k = acos(-g / (2 * i));
			m = cos(k / 3);
			n = sqrt(3.0)*sin(k / 3);
			p = -b / (3 * a);
			x1 = 2 * j*m + p;
			x2 = -j*(m + n) + p;
			x3 = -j*(m - n) + p;
			cout << floor(x1) << " " << floor(x2) << " " << floor(x3) << endl;
		}
	}

	return 0;
}
int f8(const int n) {
	KajVec tab;
	fill(n, tab);


	double A;
	double B;
	double C = 0;
	for (int i = 1; i <= tab[0]; i++) {

		A = (i * ((i + 1)* (i + 1)));
		B = C + A;
		C = B;
	}

	cout << fixed << setprecision(0) << floor(B) << endl;
	return 0;

}

void f9(const int n) {
	KajVec tab;
	fill(n, tab);
	
	for (int i = 0; i < n; i++) {
		unsigned int k = 0;
		while (tab[i])
		{
			tab[i] &= (tab[i] - 1);
			k++;
		}
		cout << k << " ";
	}
	


	cout << "\n";
}



int main()
{

	int subprogram, n;
	while (cin >> subprogram >> n)
	{
		switch (subprogram)
		{
		case 0:
		{
			KajVec tab;
			f0(n, 1, tab);
		}
		break;
		case 1:
			f1(n);
			break;
		case 2:
		{
			KajVec tab;
			cout << floor(f2(n, tab, 1)) << endl;
		}
		break;
		case 3:
			f3(n);
			break;
		case 4: 
			f4(n);
			break;
		case 5:
			f5(n);
			break;
		case 6:
			f6(n);
			break;
		case 7:
			f7(n);
			break;
		case 8:
			f8(n);
			break;
		case 9:
			f9(n);
			break;
		}
	};
	return 0;
}
