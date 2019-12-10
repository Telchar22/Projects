using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace HealthySnakeGame
{
    public partial class frmSnake : Form
    {
        Random rand;
        enum Fields
        {
            Free,
            Snake,
            Bonus,
            NotBonus
        };
        enum Directions
        {
            Up,
            Down,
            Left,
            Right
        };

        struct SnakeCoordinates
        {
            public int x;
            public int y;

        }

        Fields[,] fields;  //obszary planszy, array 2D
        SnakeCoordinates[] snakeXY; //array połozenia weza
        int snakeLength;  //dlugosc weza
        Directions directions; //przetrzymuje kierunek poruszanai sie weza
        Graphics g; //klasa do "malowania" w oknie



        public frmSnake()
        {
            InitializeComponent();
            fields = new Fields[11, 11]; // plansza 12x12 inicializacja
            snakeXY = new SnakeCoordinates[100];
            rand = new Random();
        }

        private void frmSnake_Load(object sender, EventArgs e)
        {
            PicGameBoard.Image = new Bitmap(420, 420); //tworzymy bit mape
            g = Graphics.FromImage(PicGameBoard.Image); //metoda klasy g FromImage - uzwyamy bitmape
            g.Clear(Color.White); // biale tlo w oknie

            for (int i = 1; i <= 10; i++)
            {
                //gora dol malowanie murka
                g.DrawImage(imgList.Images[8], i * 35, 0);//obrazki 35pix, wiec malujemy murek co 35 pix
                g.DrawImage(imgList.Images[8], i * 35, 385);

            }

            for (int i = 0; i <= 11; i++)
            {
                g.DrawImage(imgList.Images[8], 0, i * 35); ////obrazki 35pix, wiec malujemy murek co 35 pix
                g.DrawImage(imgList.Images[8], 385, i * 35);

            }
            //ustawiamy weza na srodku
            snakeXY[0].x = 5;
            snakeXY[0].y = 5;
            snakeXY[1].x = 5;
            snakeXY[1].y = 6;

            g.DrawImage(imgList.Images[7], 5 * 35, 5 * 35); //glowa
            g.DrawImage(imgList.Images[6], 5 * 35, 6 * 35); //1 segemnt

            fields[5, 5] = Fields.Snake;
            fields[5, 6] = Fields.Snake;  //rysowanie na planszy
           

            directions = Directions.Up;
            snakeLength = 2;

            for (int i = 0; i < 4; i++)
            {
                if (i % 2 == 0)
                {
                    Bonus();
                }
                else
                {
                    NotBonus();
                }
            }
            
            

        }

        private void Bonus()
        {
            int x, y;
            var imgIndex = rand.Next(0, 3);
            
            //generowane sa nowe koordynaty jak sie nakladaja z polozeniem weza albo murku
            do
            {
                x = rand.Next(1, 10);
                y = rand.Next(1, 10);
            }
            while (fields[x, y] != Fields.Free);

            fields[x, y] = Fields.Bonus; //przypisanie obszaru do bonusu
            g.DrawImage(imgList.Images[imgIndex], x * 35, y * 35); //rysowanie jedzenia na planszy  
        }
        private void NotBonus()
        {
            int x, y;
            var imgIndex = rand.Next(3, 6);

            //generowane sa nowe koordynaty jak sie nakladaja z polozeniem weza albo murku
            do
            {
                x = rand.Next(1, 10);
                y = rand.Next(1, 10);
            }
            while (fields[x, y] != Fields.Free);

            fields[x, y] = Fields.NotBonus; //przypisanie obszaru do notbonusu
            g.DrawImage(imgList.Images[imgIndex], x * 35, y * 35); //rysowanie jedzenia na planszy  
        }

        private void frmSnake_KeyDown(object sender, KeyEventArgs e)
        {
            switch (e.KeyCode)
            {
                case Keys.Up:
                    directions = Directions.Up;
                    break;
                case Keys.Down:
                    directions = Directions.Down;
                    break;
                case Keys.Left:
                    directions = Directions.Left;
                    break;
                case Keys.Right:
                    directions = Directions.Right;
                    break;
            }
        }

        private void GameOver()
        {
            timer.Enabled = false;
            MessageBox.Show("GAME OVER");
        }

        private void Timer_Tick(object sender, EventArgs e)
        {
            //usuwanie ostatniego segmentu
            g.FillRectangle(Brushes.White, snakeXY[snakeLength - 1].x * 35,
                snakeXY[snakeLength - 1].y * 35, 35, 35);
            //czyszczenie miesjca planszy gdzie był ostani segment
            fields[snakeXY[snakeLength - 1].x, snakeXY[snakeLength - 1].y] = Fields.Free;

            //przesniece seg 2 na poz 1
            for (int i = snakeLength; i >= 1; i--)
            {
                snakeXY[i].x = snakeXY[i - 1].x;
                snakeXY[i].y = snakeXY[i - 1].y;
            }

            g.DrawImage(imgList.Images[6], snakeXY[0].x * 35, snakeXY[0].y * 35);

           
            switch (directions)
            {
                case Directions.Up:
                    snakeXY[0].y = snakeXY[0].y - 1;
                    break;
                case Directions.Down:
                    snakeXY[0].y = snakeXY[0].y + 1;
                    break;
                case Directions.Left:
                    snakeXY[0].x = snakeXY[0].x - 1;
                    break;
                case Directions.Right:
                    snakeXY[0].x = snakeXY[0].x + 1;
                    break;
            }

            //sprawdzanei czy waz uderzyl w sciane
            if (snakeXY[0].x < 1 || snakeXY[0].x > 10 || snakeXY[0].y < 1 || snakeXY[0].y > 10)
            {
                GameOver();
                PicGameBoard.Refresh();
                return;
            }

            //sprawdzanei czy ugryzło sie samego siebie
            if (fields[snakeXY[0].x, snakeXY[0].y] == Fields.Snake)
            {
                GameOver();
                PicGameBoard.Refresh();
                return;
            }
            
            //sporawdzanei czy waz zjadl bonus
            if (fields[snakeXY[0].x, snakeXY[0].y] == Fields.Bonus)
            {

            
            //ddoanei seg
            g.DrawImage(imgList.Images[6], snakeXY[snakeLength].x * 35,
                snakeXY[snakeLength].y * 35);
            //dodanie obszaru planszy do weza z nowym segmentem
            fields[snakeXY[snakeLength].x, snakeXY[snakeLength].y] = Fields.Snake;
            snakeLength++;

                if (snakeLength < 96)
                    Bonus();

                this.Text = "Snake - score: " + snakeLength;
            }
            if (fields[snakeXY[0].x, snakeXY[0].y] == Fields.NotBonus)
            {
             g.FillRectangle(Brushes.White, snakeXY[snakeLength - 1].x * 35, snakeXY[snakeLength - 1].y * 35, 35, 35);
                //czyszczenie miesjca planszy gdzie był ostani segment
              
             fields[snakeXY[snakeLength - 1].x, snakeXY[snakeLength - 1].y] = Fields.Free;
             snakeLength--;

                if (snakeLength < 96)
                    NotBonus();

                this.Text = "Snake - score: " + snakeLength;
            }
            if (snakeLength == 1)
            {
                GameOver();
                PicGameBoard.Refresh();
                return;
            }

           
            g.DrawImage(imgList.Images[7], snakeXY[0].x * 35, snakeXY[0].y * 35);
            fields[snakeXY[0].x, snakeXY[0].y] = Fields.Snake;
            PicGameBoard.Refresh();

        }

    }
}
