using System;
using System.Collections;
using System.Collections.Generic;

namespace ConsoleApp3
{
    public enum Language
    {
        eng,
        pol,
        ger
    }

    public abstract class Test
    {
        public Language lang;
        public Hashtable NAM = new Hashtable();

        public Test()
        {
        }

        public Test(Language lan)
        {
            lang = lan;
        }

        public abstract void Describe();
    }

    public class MathTest : Test
    {
        public override void Describe()
        {
            Console.WriteLine("Solve:");
        }
    }

    public class LanguageTest : Test
    {
        public LanguageTest(Language lan) : base(lan)
        {
        }

        public override void Describe()
        {
            Console.WriteLine("Listen:");
        }
    }

    internal class Program
    {

        public delegate void Student(string name);
        private static void Main(string[] args)
        {

            var tests = new List<Test>
            {
                new MathTest(),
                new LanguageTest(Language.eng),
                new LanguageTest(Language.ger)
            };
            tests[0].NAM.Add("Kowalski", 3);
            tests[1].NAM.Add("Kowalski", 2);
            tests[2].NAM.Add("Kowalski", 1);

            tests[0].NAM.Add("Jablonski", 4);
            tests[1].NAM.Add("Jablonski", 3);
            tests[2].NAM.Add("Jablonski", 4);

            tests[0].NAM.Add("Grud", 5);
            tests[1].NAM.Add("Grud", 3);
            tests[2].NAM.Add("Grud", 3);

            Console.WriteLine("Name:");
            string name = Console.ReadLine();

            Console.WriteLine("a or c");
            string deleg = Console.ReadLine();

            Student student = null;

            switch (deleg)
            {
                case "a":

                    {
                        int sum = 0;

                        foreach (var elements in tests)
                        {
                            if (elements.NAM.Contains(name))
                            {
                                var NAM = Int32.Parse(elements.NAM[name].ToString());
                                Console.WriteLine($"Results: {NAM}");
                                sum += NAM;
                            }
                        }
                        Console.WriteLine($"Avr mark for: {name} is : {sum / 3.0} (and sum is: {sum})");
                        
                    }
                    break;
                case "c":
                    {
                        foreach (var elements in tests)
                        {
                            if (elements.NAM.Contains(name))
                            {
                                var NAM = Int32.Parse(elements.NAM[name].ToString());
                                if (NAM < 2)
                                    Console.WriteLine($"Ur mark form {elements.GetType()}: is {NAM}, SEE YOU NEXT YEAR!");
                            }
                        }
                    }
                    break;
                default:
                    student = new Student(x => { Console.WriteLine($"RUMTMTUM: {name}"); });
                    break;
            }








            

            
        }

    }
}
