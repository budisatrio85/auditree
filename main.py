from fastapi.responses import JSONResponse
from fastapi import FastAPI, File
from starlette.requests import Request
import uvicorn
from pydantic import BaseModel
import pickle
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import io
from victorinox import victorinox
from glob import glob
import os
import re

tool=victorinox()
population1_dict={}
population2_dict={}
population_root_path=r"corpus/population"
population_files=glob(os.path.join(population_root_path, "**/*.txt"),recursive=True)

for f in population_files:
    id=os.path.split(f)[-1].replace(".txt","")
    (points,violations)=tool.extract_benford(currency_path=f,
                         digs=1)
    population1_dict[id]=(points,violations)
    (points2, violations2) = tool.extract_benford(currency_path=f,
                                                digs=2)
    population2_dict[id] = (points2, violations2)

individu1_dict={}
individu2_dict={}
individu_root_path=r"corpus/individu"
individu_files=glob(os.path.join(individu_root_path, "**/*.txt"),recursive=True)

pp=re.compile("\d+")
for f in individu_files:
    try:
        folder,fn=os.path.split(f)
        company=str(folder).split("/")[-1]
        year=pp.findall(fn)[0]
        id=company+"_"+year
        (points,violations)=tool.extract_benford(currency_path=f,
                             digs=1)
        individu1_dict[id]=(points,violations)
        (points2, violations2) = tool.extract_benford(currency_path=f,
                                                    digs=2)
        individu2_dict[id] = (points2, violations2)
    except Exception as e:
        print("Error file {}:{}".format(f,str(e)))
        continue

class Violation(BaseModel):
    first_digits: list
    expected: list
    found: list
    z_score: list



class Points(BaseModel):
    first_digits: list
    counts:list
    found:list
    expected:list


class BenfordModel(BaseModel):
    status: str
    points: dict
    violations: dict


# m = FooBarModel(banana=3.14, foo='hello', bar={'whatever': 123})
class Benford_Item(BaseModel):
    status: str
    val: dict


app = FastAPI()
@app.post("/population_benford")
async def population_benford(request: Request):#,
                 #pdf: bytes = File(...)):

    init_result = {
        'status': 'error',
        "val": {}
    }
    result = Benford_Item(**init_result)
    if request.method == "POST":
        try:
            form = await request.form()
            id=form["id"]
            digits = int(form["digits"])
            if digits==1:
                if id in population1_dict:
                    (points,violations)=population1_dict[id]
                    result=BenfordModel(status="ok",
                                        points={"expected":points.Expected,"found":points.Found},
                                        violations={"expected":violations.Expected,"found":violations.Found})
                else:
                    result.status="not-found"
                    result.val="id not found"
            else:
                if id in population2_dict:
                    (points,violations)=population2_dict[id]
                    result = BenfordModel(status="ok",
                                          points={"expected": points.Expected,
                                                  "found": points.Found},
                                          violations={"expected": violations.Expected,
                                                      "found": violations.Found})

                else:
                    result.status="not-found"
                    result.val="id not found"
            # stream = io.BytesIO(pdf)
        except Exception as e:
            result.status="error"
            result.val=str(e)
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)


@app.post("/individu_benford")
async def individu_benford(request: Request):#,
                 #pdf: bytes = File(...)):
    init_result = {
         'status': 'error',
         "val": {}
     }
    result = Benford_Item(**init_result)
    if request.method == "POST":
        try:
            form = await request.form()
            id=form["id"]
            digits = int(form["digits"])
            if digits==1:
                if id in individu1_dict:
                    (points,violations)=individu1_dict[id]
                    result=BenfordModel(status="ok",
                                        points={"expected":points.Expected,"found":points.Found},
                                        violations={"expected":violations.Expected,"found":violations.Found})
                else:
                    result.status="not-found"
                    result.val="id not found"
            else:
                if id in individu2_dict:
                    (points,violations)=individu2_dict[id]
                    result = BenfordModel(status="ok",
                                          points={"expected": points.Expected,
                                                  "found": points.Found},
                                          violations={"expected": violations.Expected,
                                                      "found": violations.Found})

                else:
                    result.status="not-found"
                    result.val="id not found"
            # stream = io.BytesIO(pdf)
        except Exception as e:
            result.status="error"
            result.val=str(e)
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)


@app.post("/test")
async def verify(request: Request):
    init_result = {
        'status': 'error',
        "val": "not registered"
    }
    result = Benford_Item(**init_result)
    result.status="ok"
    result.val="jos"
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)



