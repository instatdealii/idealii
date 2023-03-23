// ---------------------------------------------------------------------
//
// Copyright (C) 2022 - 2023 by the ideal.II authors
//
// This file is part of the ideal.II library.
//
// The ideal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of ideal.II.
//
// ---------------------------------------------------------------------

#include <ideal.II/base/time_iterator.hh>

namespace idealii
{
    template<int dim>
    TimeIteratorCollection<dim>::TimeIteratorCollection ()
    {
        tria.it_collection = std::vector<slab::TriaIterator<dim>*> ();
        tria.obj_collection =
                std::vector<spacetime::Triangulation<dim>*> ();
        par_dist_tria.it_collection = std::vector<
                slab::parallel::distributed::TriaIterator<dim>*> ();
        par_dist_tria.obj_collection = std::vector<
                spacetime::parallel::distributed::Triangulation<dim>*> ();
        dof.it_collection = std::vector<slab::DoFHandlerIterator<dim>*> ();
        dof.obj_collection = std::vector<spacetime::DoFHandler<dim>*> ();
        vector_double.it_collection = std::vector<
                slab::VectorIterator<double>*> ();
        vector_double.obj_collection = std::vector<
                spacetime::Vector<double>*> ();
        trilinos_vector.it_collection = std::vector<
                slab::TrilinosVectorIterator*> ();
        trilinos_vector.obj_collection = std::vector<
                spacetime::TrilinosVector*> ();
    }

    template<int dim>
    void TimeIteratorCollection<dim>::add_iterator (
            slab::TriaIterator<dim> *it ,
            spacetime::Triangulation<dim> *obj )
    {
        tria.it_collection.push_back ( it );
        tria.obj_collection.push_back ( obj );
    }

#ifdef DEAL_II_WITH_MPI
template<int dim>
void TimeIteratorCollection<dim>::add_iterator (
        slab::parallel::distributed::TriaIterator<dim> *it ,
        spacetime::parallel::distributed::Triangulation<dim> *obj )
{
    par_dist_tria.it_collection.push_back ( it );
    par_dist_tria.obj_collection.push_back ( obj );
}
#endif
template<int dim>
void TimeIteratorCollection<dim>::add_iterator (
        slab::DoFHandlerIterator<dim> *it ,
        spacetime::DoFHandler<dim> *obj )
{
    dof.it_collection.push_back ( it );
    dof.obj_collection.push_back ( obj );
}

template<int dim>
void TimeIteratorCollection<dim>::add_iterator (
        slab::VectorIterator<double> *it ,
        spacetime::Vector<double> *obj )
{
    vector_double.it_collection.push_back ( it );
    vector_double.obj_collection.push_back ( obj );
}

template<int dim>
void TimeIteratorCollection<dim>::add_iterator (
        slab::TrilinosVectorIterator *it ,
        spacetime::TrilinosVector *obj )
{
    trilinos_vector.it_collection.push_back ( it );
    trilinos_vector.obj_collection.push_back ( obj );
}
template<int dim>
void TimeIteratorCollection<dim>::increment ()
{
    for ( unsigned int i = 0 ; i < tria.it_collection.size () ; i++ )
    {
        Assert( *tria.it_collection[i] != tria.obj_collection[i]->end () ,
                dealii::ExcIteratorPastEnd () );
        ++( *tria.it_collection[i] );
    }

#ifdef DEAL_II_WITH_MPI
for ( unsigned int i = 0 ; i < par_dist_tria.it_collection.size () ;
        i++ )
{
    Assert( *par_dist_tria.it_collection[i] != par_dist_tria.obj_collection[i]->end () ,
            dealii::ExcIteratorPastEnd () );
    ++( *par_dist_tria.it_collection[i] );
}
#endif
for ( unsigned int i = 0 ; i < dof.it_collection.size () ; i++ )
{
    Assert( *dof.it_collection[i] != dof.obj_collection[i]->end () ,
            dealii::ExcIteratorPastEnd () );
    ++( *dof.it_collection[i] );
}
for ( unsigned int i = 0 ; i < vector_double.it_collection.size () ;
        i++ )
{
    Assert( *vector_double.it_collection[i] != vector_double.obj_collection[i]->end () ,
            dealii::ExcIteratorPastEnd () );
    ++( *vector_double.it_collection[i] );
}
for ( unsigned int i = 0 ;
        i < trilinos_vector.it_collection.size () ; i++ )
{
    Assert( *trilinos_vector.it_collection[i] != trilinos_vector.obj_collection[i]->end () ,
            dealii::ExcIteratorPastEnd () );
    ++( *trilinos_vector.it_collection[i] );
}
}

template<int dim>
void TimeIteratorCollection<dim>::decrement ()
{
    for ( unsigned int i = 0 ; i < tria.it_collection.size () ; i++ )
    {
        Assert( *tria.it_collection[i] != std::prev (
                tria.obj_collection[i]->begin () ) ,
                dealii::ExcIteratorPastEnd () );
        --( *tria.it_collection[i] );
    }

#ifdef DEAL_II_WITH_MPI
for ( unsigned int i = 0 ; i < par_dist_tria.it_collection.size () ;
        i++ )
{
    Assert( *par_dist_tria.it_collection[i] != std::prev (
            par_dist_tria.obj_collection[i]->begin () ) ,
            dealii::ExcIteratorPastEnd () );
    --( *par_dist_tria.it_collection[i] );
}
#endif
for ( unsigned int i = 0 ; i < dof.it_collection.size () ; i++ )
{
    Assert( *dof.it_collection[i] != std::prev (
            dof.obj_collection[i]->begin () ) ,
            dealii::ExcIteratorPastEnd () );
    --( *dof.it_collection[i] );
}
for ( unsigned int i = 0 ; i < vector_double.it_collection.size () ;
        i++ )
{
    Assert( *vector_double.it_collection[i] != std::prev (
            vector_double.obj_collection[i]->begin () ) ,
            dealii::ExcIteratorPastEnd () );
    --( *vector_double.it_collection[i] );
}
#ifdef DEAL_II_WITH_TRILINOS
#ifdef DEAL_II_WITH_MPI
for ( unsigned int i = 0 ;
        i < trilinos_vector.it_collection.size () ; i++ )
{
    Assert( *trilinos_vector.it_collection[i] != std::prev (
            trilinos_vector.obj_collection[i]->begin () ) ,
            dealii::ExcIteratorPastEnd () );
    --( *trilinos_vector.it_collection[i] );
}
#endif
#endif
}

template<int dim>
bool TimeIteratorCollection<dim>::at_end ()
{
    bool res = false;
    for ( unsigned int i = 0 ; i < tria.it_collection.size () ; i++ )
    {
        if ( *tria.it_collection[i] == tria.obj_collection[i]->end () )
        {
            res = true;
        }
    }

#ifdef DEAL_II_WITH_MPI
for ( unsigned int i = 0 ; i < par_dist_tria.it_collection.size () ;
        i++ )
{
    if ( *par_dist_tria.it_collection[i] == par_dist_tria.obj_collection[i]->end () )
    {
        res = true;
    }
}
#endif
for ( unsigned int i = 0 ; i < dof.it_collection.size () ; i++ )
{
    if ( *dof.it_collection[i] == dof.obj_collection[i]->end () )
    {
        res = true;
    }
}
for ( unsigned int i = 0 ; i < vector_double.it_collection.size () ;
        i++ )
{
    if ( *vector_double.it_collection[i] == vector_double.obj_collection[i]->end () )
    {
        res = true;
    }
}
#ifdef DEAL_II_WITH_TRILINOS
#ifdef DEAL_II_WITH_MPI
for ( unsigned int i = 0 ;
        i < trilinos_vector.it_collection.size () ; i++ )
{
    if ( *trilinos_vector.it_collection[i] == trilinos_vector.obj_collection[i]->end () )
    {
        res = true;
    }
}
#endif
#endif
return res;
}

template<int dim>
bool TimeIteratorCollection<dim>::before_begin ()
{
    bool res = false;
    for ( unsigned int i = 0 ; i < tria.it_collection.size () ; i++ )
    {
        if ( *tria.it_collection[i] == std::prev (
                tria.obj_collection[i]->begin () ) )
        {
            res = true;
        }
    }
#ifdef DEAL_II_WITH_MPI
for ( unsigned int i = 0 ; i < par_dist_tria.it_collection.size () ;
        i++ )
{
    if ( *par_dist_tria.it_collection[i] == std::prev (
            par_dist_tria.obj_collection[i]->begin () ) )
    {
        res = true;
    }
}
#endif
for ( unsigned int i = 0 ; i < dof.it_collection.size () ; i++ )
{
    if ( *dof.it_collection[i] == std::prev (
            dof.obj_collection[i]->begin () ) )
    {
        res = true;
    }
}
for ( unsigned int i = 0 ; i < vector_double.it_collection.size () ;
        i++ )
{
    if ( *vector_double.it_collection[i] == std::prev (
            vector_double.obj_collection[i]->begin () ) )
    {
        res = true;
    }
}
#ifdef DEAL_II_WITH_TRILINOS
#ifdef DEAL_II_WITH_MPI
for ( unsigned int i = 0 ;
        i < trilinos_vector.it_collection.size () ; i++ )
{
    if ( *trilinos_vector.it_collection[i] == std::prev (
            trilinos_vector.obj_collection[i]->begin () ) )
    {
        res = true;
    }
}
#endif
#endif
return res;
}
}

#include "time_iterator.inst"

